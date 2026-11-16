"""Train an xG CNN with early stopping on the validation log loss.

    python -m src.train --model small_cnn
    python -m src.train --model resnet18
    python -m src.train --model resnet18 --subset 2000 --epochs 3      # smoke test
    python -m src.train --bench-loader                                 # DataLoader speed

Outputs go to models/<run_name>/: train.log, best.pt, last.pt, history.csv, config.json, val_predictions.csv,
and results/plots/training_curves_<run_name>.png. The test split is never touched here.
"""
from __future__ import annotations

import os

# DataLoader workers are separate processes that inherit these variables. One BLAS/OpenMP
# thread per worker: rendering is already parallel across workers, and the default
# (one thread pool + buffers per core in every worker) runs Windows out of commit memory.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config as C
from src.dataset import PhaseBatchSampler, ShotDataset, load_shots
from src.evaluate import compute_metrics, format_table, metrics_by_group, reference_predictions
from src.models import build_model, count_parameters
from src.utils import get_device, seed_everything, tee_output

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}


# ---------------------------------------------------------------------------
# Data: one DataLoader with persistent workers serves every phase
# ---------------------------------------------------------------------------
class ShotLoader:
    """
    Train and val shots in one ShotDataset behind a PhaseBatchSampler, so the
    worker processes are started once per run (Windows spawns them from scratch,
    ~10 s each) and never more than `num_workers` exist at a time.

    phases: "train" (shuffled, mirrored, drop_last), "bn" (a few training batches
    for BatchNorm recalibration) and "val" (ordered, no augmentation).
    """

    def __init__(self, train_df, val_df, batch_size: int, num_workers: int,
                 max_train_batches=None, bn_batches: int = C.BN_RECAL_BATCHES, seed: int = C.SEED):
        both = pd.concat([train_df, val_df], ignore_index=True)
        self.n_train, self.n_val = len(train_df), len(val_df)
        tr_idx = np.arange(self.n_train)
        self.sampler = PhaseBatchSampler({
            "train": (tr_idx, True, max_train_batches),
            "bn": (tr_idx, True, bn_batches),
            "val": (np.arange(self.n_train, len(both)), False, None),
        }, batch_size, seed)
        self.loader = DataLoader(
            ShotDataset(both, image_dtype=torch.float16),        # cast to fp32 on the GPU
            batch_sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def __call__(self, phase: str):
        self.sampler.phase = phase
        return self.loader


def benchmark_loader(train_df, batch_size: int, worker_counts=(0, 6), n_batches: int = 60) -> None:
    """Time worker start-up and steady-state speed of the training DataLoader alone."""
    for nw in worker_counts:
        loaders = ShotLoader(train_df, train_df.iloc[:0], batch_size, nw)
        t0 = time.perf_counter()
        it = iter(loaders("train"))
        next(it)
        t_first = time.perf_counter() - t0                 # worker start-up + first batch
        t0 = time.perf_counter()
        for _ in range(n_batches):
            next(it)
        dt = (time.perf_counter() - t0) / n_batches
        per_epoch = dt * (len(train_df) // batch_size)
        print(f"num_workers={nw}: start-up + first batch {t_first:.1f}s, {dt * 1000:.0f} ms/batch "
              f"({dt / batch_size * 1000:.2f} ms/sample) -> data only ~{per_epoch:.0f}s per epoch")
        del it, loaders


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def autocast(device, amp_dtype):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


def train_one_epoch(model, loader, optimizer, loss_fn, device, amp_dtype, scaler):
    model.train()
    total, n = 0.0, 0
    for img, extra, y, _ in loader:
        img = img.to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
        extra, y = extra.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            logits = model(img, extra)
        loss = loss_fn(logits.float(), y)           # loss in fp32
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, amp_dtype, n: int, offset: int = 0) -> np.ndarray:
    """P(goal) for the `n` samples that `loader` yields (dataset indices offset..offset+n-1)."""
    model.eval()
    out = np.full(n, np.nan)
    for img, extra, _, idx in loader:
        img = img.to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
        with autocast(device, amp_dtype):
            logits = model(img, extra.to(device, non_blocking=True))
        out[idx.numpy() - offset] = torch.sigmoid(logits.float()).cpu().numpy()
    assert not np.isnan(out).any(), "some samples were not predicted"
    return out


@torch.no_grad()
def recalibrate_bn(model, loader, device, amp_dtype) -> None:
    """
    'Precise BN': recompute BatchNorm running statistics with the current, frozen
    weights (cumulative average over the batches of `loader`).

    The running averages collected during training lag behind the quickly changing
    weights; in the step-2 smoke test this shifted the mean val prediction from the
    true goal rate (~0.10) to 0.03-0.05 in eval() mode, while batch statistics gave 0.11.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                               # cumulative moving average
    model.train()
    for m in model.modules():                            # keep the forward deterministic
        if isinstance(m, nn.Dropout):
            m.eval()
    for img, extra, _, _ in loader:
        img = img.to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
        with autocast(device, amp_dtype):
            model(img, extra.to(device, non_blocking=True))
    for bn, mom in zip(bns, momenta):
        bn.momentum = mom
    model.eval()


def plot_history(hist: pd.DataFrame, run_name: str, sb_val: dict, save_path) -> None:
    import matplotlib                                    # lazy: workers re-import this module
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train (BCE, augmented)")
    axes[0].plot(hist["epoch"], hist["val_log_loss"], "s-", label="val log loss")
    axes[0].axhline(sb_val["log_loss"], color="grey", ls="--", label="StatsBomb xG (val)")
    axes[0].set_title("Loss")
    axes[1].plot(hist["epoch"], hist["val_brier"], "s-", color="C1")
    axes[1].axhline(sb_val["brier"], color="grey", ls="--")
    axes[1].set_title("Val Brier")
    axes[2].plot(hist["epoch"], hist["val_roc_auc"], "s-", color="C2")
    axes[2].axhline(sb_val["roc_auc"], color="grey", ls="--")
    axes[2].set_title("Val ROC-AUC")
    best = hist.loc[hist["val_log_loss"].idxmin(), "epoch"]
    for ax in axes:
        ax.axvline(best, color="C3", alpha=0.3)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run_name}: training curves (red line = best epoch)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def train(args) -> None:
    seed_everything(C.SEED)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    amp_dtype = AMP_DTYPES[args.amp]
    mcfg = C.MODEL_CONFIGS[args.model]
    run_name = args.run_name or args.model
    run_dir = C.MODELS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_shots("train"), load_shots("val")
    if args.subset:
        train_df = train_df.sample(min(args.subset, len(train_df)), random_state=C.SEED).reset_index(drop=True)
        val_df = val_df.sample(min(args.subset, len(val_df)), random_state=C.SEED).reset_index(drop=True)
    prior = float(train_df["is_goal"].mean())
    loaders = ShotLoader(train_df, val_df, args.batch_size, args.num_workers, args.max_batches)
    n_tr, n_va = len(train_df), len(val_df)

    model = build_model(args.model, prior=prior).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.param_groups(mcfg["lr"], mcfg["backbone_lr"]),
                                  weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=C.LR_PLATEAU_FACTOR, patience=C.LR_PLATEAU_PATIENCE)
    loss_fn = nn.BCEWithLogitsLoss()                 # plain BCE: no class weights (calibration)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    sb_val = compute_metrics(val_df["is_goal"], val_df["statsbomb_xg"])
    run_cfg = {"model": args.model, **mcfg, "batch_size": args.batch_size, "amp": args.amp,
               "max_epochs": args.epochs, "patience": args.patience, "weight_decay": C.WEIGHT_DECAY,
               "bn_recal_batches": C.BN_RECAL_BATCHES, "subset": args.subset, "seed": C.SEED,
               "n_train": len(train_df), "n_val": len(val_df), "prior": prior, "n_params": count_parameters(model)}
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))
    print(f"{run_name}: {run_cfg['n_params'] / 1e6:.2f} M params | train {len(train_df)} | val {len(val_df)} "
          f"| device {device} | amp {args.amp}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, bad_epochs = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(model, loaders("train"), optimizer, loss_fn, device, amp_dtype, scaler)
        t_train = time.perf_counter() - t0
        recalibrate_bn(model, loaders("bn"), device, amp_dtype)
        p_val = predict(model, loaders("val"), device, amp_dtype, n_va, offset=n_tr)
        m = compute_metrics(val_df["is_goal"], p_val)
        scheduler.step(m["log_loss"])
        row = {"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in m.items() if k != "n"},
               "lr": optimizer.param_groups[0]["lr"], "time_train_s": t_train,
               "time_total_s": time.perf_counter() - t0,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0}
        history.append(row)

        ckpt = {"model_name": args.model, "state_dict": model.state_dict(), "epoch": epoch,
                "prior": prior, "val_metrics": m, "config": run_cfg}
        torch.save(ckpt, run_dir / "last.pt")
        improved = m["log_loss"] < best - 1e-5
        if improved:
            best, bad_epochs = m["log_loss"], 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad_epochs += 1
        print(f"ep {epoch:2d} | train {tr_loss:.4f} | val logloss {m['log_loss']:.4f} brier {m['brier']:.4f} "
              f"auc {m['roc_auc']:.4f} mean_p {m['mean_pred']:.3f} | lr {row['lr']:.1e} | "
              f"{row['time_total_s']:.0f}s | vram {row['peak_vram_gb']:.2f} GB{' *' if improved else ''}")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if bad_epochs >= args.patience:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    hist = pd.DataFrame(history)
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_history(hist, run_name, sb_val, C.PLOTS_DIR / f"training_curves_{run_name}.png")

    # Final report with the best checkpoint
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    p_val = predict(model, loaders("val"), device, amp_dtype, n_va, offset=n_tr)
    pd.DataFrame({"shot_id": val_df["shot_id"], "p_goal": p_val}).to_csv(run_dir / "val_predictions.csv",
                                                                           index=False)
    res = {run_name: compute_metrics(val_df["is_goal"], p_val)}
    res.update({k: compute_metrics(val_df["is_goal"], p)
                for k, p in reference_predictions(val_df, prior).items()})
    print(f"\nbest epoch {ckpt['epoch']} -- validation split:")
    print(format_table(res))
    print("note: the epoch (and any model choice) is selected on val, so the CNN looks slightly better\n"
          "here than it is; StatsBomb xG was not tuned on this split. The real comparison is on the\n"
          "test split in step 3.")
    print(f"\n{run_name} on val by gender:")
    print(metrics_by_group(val_df, p_val)[["n", "log_loss", "brier", "roc_auc", "goal_rate", "mean_pred"]]
          .astype(float).round(4).to_string())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small_cnn", choices=list(C.MODEL_CONFIGS))
    ap.add_argument("--run-name", default=None, help="defaults to the model name")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--amp", default=C.AMP_DTYPE, choices=list(AMP_DTYPES))
    ap.add_argument("--subset", type=int, default=None, help="use N random train and val shots (smoke test)")
    ap.add_argument("--max-batches", type=int, default=None, help="cap training batches per epoch")
    ap.add_argument("--bench-loader", action="store_true", help="only time the DataLoader")
    ap.add_argument("--bench-workers", type=int, nargs="+", default=[0, 6],
                    help="worker counts tried by --bench-loader")
    return ap.parse_args()


if __name__ == "__main__":
    # Guard is required on Windows: DataLoader workers are spawned processes that re-import this module.
    args = parse_args()
    if args.bench_loader:
        benchmark_loader(load_shots("train"), args.batch_size, args.bench_workers)
    else:
        # everything printed (and any traceback) also goes to models/<run>/train.log
        with tee_output(C.MODELS_DIR / (args.run_name or args.model) / "train.log"):
            train(args)
