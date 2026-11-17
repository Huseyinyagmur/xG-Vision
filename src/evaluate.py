"""Evaluation shared by every model (CNNs, baselines, StatsBomb xG).

    python -m src.evaluate                 # reference metrics (StatsBomb xG, constant) on val and test
    python -m src.evaluate --split val     # full report on val (used while developing)
    python -m src.evaluate --split test    # full report on test: run once, after EVAL_MODELS is fixed

The full report writes:
    models/predictions_<split>.csv         every model's P(goal) per shot (not committed)
    results/metrics_<split>.md             metric tables, seeds, TTA, bootstrap CIs, calibration
    results/plots/calibration_<split>.png  (+ _by_gender) reliability diagrams
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src import config as C

EPS = 1e-7
METRICS = ("log_loss", "brier", "roc_auc")


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, p_pred) -> dict:
    """Log loss, Brier and ROC-AUC (+ counts and mean prediction for a calibration-in-the-large check)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_pred, dtype=np.float64), EPS, 1 - EPS)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "goal_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
    }


def metrics_by_group(df: pd.DataFrame, p_pred, group_col: str = "gender") -> pd.DataFrame:
    """Metrics on the whole frame ('all') plus one row per value of `group_col`."""
    p = np.asarray(p_pred)
    rows = {"all": compute_metrics(df["is_goal"], p)}
    for g, idx in df.groupby(group_col).indices.items():
        rows[g] = compute_metrics(df["is_goal"].to_numpy()[idx], p[idx])
    return pd.DataFrame(rows).T


def reference_predictions(df: pd.DataFrame, train_goal_rate: float) -> dict[str, np.ndarray]:
    """Predictions of the reference models that need no training."""
    return {
        "statsbomb_xg": df["statsbomb_xg"].to_numpy(np.float64),
        "constant": np.full(len(df), train_goal_rate),
    }


def format_table(results: dict[str, dict], cols=("log_loss", "brier", "roc_auc", "mean_pred")) -> str:
    return pd.DataFrame(results).T[list(cols)].astype(float).round(4).to_string()


# ---------------------------------------------------------------------------
# Predictions of every model on one split
# ---------------------------------------------------------------------------
def available_runs(runs) -> list[str]:
    return [r for r in dict.fromkeys(runs) if (C.MODELS_DIR / r / "best.pt").exists()]


def predict_cnns(runs, df: pd.DataFrame, mirror: bool = False, batch_size: int = 512) -> dict[str, np.ndarray]:
    """
    P(goal) of several trained runs in one pass over `df` (each tensor is rendered once).
    Rendering runs in the main process (~1.2 ms per shot), so no worker processes are needed.
    `mirror=True` predicts the left/right mirrored shots (for test-time augmentation).
    """
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import ShotDataset
    from src.models import load_checkpoint
    from src.utils import get_device

    device = get_device()
    models = {r: load_checkpoint(C.MODELS_DIR / r / "best.pt", device)[0] for r in runs}
    batches = [[(i, mirror) for i in range(s, min(s + batch_size, len(df)))]
               for s in range(0, len(df), batch_size)]
    loader = DataLoader(ShotDataset(df, image_dtype=torch.float16), batch_sampler=batches, num_workers=0)
    out = {r: np.full(len(df), np.nan) for r in runs}
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu")
    with torch.no_grad(), amp:
        for img, extra, _, idx in loader:
            img = img.to(device).float().contiguous(memory_format=torch.channels_last)
            extra = extra.to(device)
            for r, m in models.items():
                out[r][idx.numpy()] = torch.sigmoid(m(img, extra).float()).cpu().numpy()
    assert all(not np.isnan(p).any() for p in out.values())
    return out


def predict_all(split: str, tta: bool = True) -> pd.DataFrame:
    """One row per shot of `split`: ids, label, gender and P(goal) of every model and baseline."""
    from src.baselines import logreg_predictions
    from src.dataset import load_shots

    df, train_df = load_shots(split), load_shots("train")
    pred = df[["shot_id", "match_id", "gender", "is_goal"]].copy()
    runs = available_runs([*C.EVAL_MODELS, *(r for g in C.SEED_GROUPS.values() for r in g)])
    missing = sorted(set(C.EVAL_MODELS) - set(runs))
    if missing:
        raise FileNotFoundError(f"no checkpoint for EVAL_MODELS {missing}")
    for r, p in predict_cnns(runs, df).items():
        pred[r] = p
    if tta:
        for r, p in predict_cnns(available_runs(C.TTA_MODELS), df, mirror=True).items():
            pred[f"{r}+tta"] = (pred[r] + p) / 2
    for k, p in reference_predictions(df, float(train_df["is_goal"].mean())).items():
        pred[k] = p
    for k, p in logreg_predictions(train_df, df).items():
        pred[k] = p
    return pred


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def metrics_table(pred: pd.DataFrame, cols) -> pd.DataFrame:
    """Long table: one row per (model, group) with group in all / female / male."""
    rows = []
    for c in cols:
        t = metrics_by_group(pred, pred[c].to_numpy())
        for g, r in t.iterrows():
            rows.append({"model": c, "group": g, **r.to_dict()})
    return pd.DataFrame(rows)


def seed_table(pred: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Mean and std over seeds of each metric, per group of runs and gender group."""
    rows = []
    for name, runs in groups.items():
        runs = [r for r in runs if r in pred]
        t = metrics_table(pred, runs)
        for g, sub in t.groupby("group"):
            row = {"model": name, "group": g, "n_seeds": len(runs)}
            for m in METRICS:
                row[f"{m}_mean"] = sub[m].mean()
                row[f"{m}_std"] = sub[m].std(ddof=1) if len(runs) > 1 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Match-level bootstrap
# ---------------------------------------------------------------------------
def _weighted_auc(order_p, order_y, groups_start, w_sorted: np.ndarray) -> np.ndarray:
    """
    ROC-AUC for many weight vectors at once. Shots are pre-sorted by score; tied
    scores form contiguous groups starting at `groups_start` and count half.
    w_sorted: (B, n) weights in that sorted order. Returns (B,).
    """
    wp = w_sorted * order_y
    wn = w_sorted * (1 - order_y)
    gp = np.add.reduceat(wp, groups_start, axis=1)
    gn = np.add.reduceat(wn, groups_start, axis=1)
    neg_below = np.cumsum(gn, axis=1) - gn
    num = (gp * (neg_below + 0.5 * gn)).sum(1)
    return num / (gp.sum(1) * gn.sum(1))


def bootstrap_by_match(pred: pd.DataFrame, cols, B: int = C.BOOTSTRAP_B, seed: int = C.BOOTSTRAP_SEED,
                       chunk: int = 250) -> dict[str, np.ndarray]:
    """
    Resample matches with replacement (all shots of a drawn match come along, with
    multiplicity). Returns {model: (B, 3) array of log loss, Brier, ROC-AUC}. The same
    resamples are used for every model, so differences between models are paired.
    """
    codes, _ = pd.factorize(pred["match_id"])
    n_matches = codes.max() + 1
    y = pred["is_goal"].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(0, n_matches, n_matches), minlength=n_matches) for _ in range(B)]
    counts = np.stack(draws).astype(np.float64)                    # (B, n_matches)

    out = {}
    for c in cols:
        p = np.clip(pred[c].to_numpy(np.float64), EPS, 1 - EPS)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        se = (p - y) ** 2
        order = np.argsort(p, kind="mergesort")
        ps, ys, cs = p[order], y[order], codes[order]
        starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
        res = np.empty((B, 3))
        for s in range(0, B, chunk):
            w = counts[s:s + chunk][:, cs]                          # (b, n) in sorted order
            wsum = w.sum(1)
            res[s:s + chunk, 0] = (w * ll[order]).sum(1) / wsum
            res[s:s + chunk, 1] = (w * se[order]).sum(1) / wsum
            res[s:s + chunk, 2] = _weighted_auc(ps, ys, starts, w)
        out[c] = res
    return out


def bootstrap_summary(pred: pd.DataFrame, boot: dict, pairs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """95% percentile CIs per model and for paired differences a - b."""
    rows = []
    for c, res in boot.items():
        point = compute_metrics(pred["is_goal"], pred[c])
        for j, m in enumerate(METRICS):
            lo, hi = np.percentile(res[:, j], [2.5, 97.5])
            rows.append({"model": c, "metric": m, "value": point[m], "ci_low": lo, "ci_high": hi})
    models = pd.DataFrame(rows)

    rows = []
    for a, b in pairs:
        if a not in boot or b not in boot:
            continue
        pa, pb = compute_metrics(pred["is_goal"], pred[a]), compute_metrics(pred["is_goal"], pred[b])
        for j, m in enumerate(METRICS):
            d = boot[a][:, j] - boot[b][:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            better = d < 0 if m != "roc_auc" else d > 0                # share of resamples where a beats b
            rows.append({"a": a, "b": b, "metric": m, "diff": pa[m] - pb[m], "ci_low": lo, "ci_high": hi,
                         "p_a_better": better.mean()})
    return models, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def reliability_bins(p, y, match_id, n_bins: int = C.CALIBRATION_BINS, B: int = 500,
                     seed: int = C.BOOTSTRAP_SEED) -> pd.DataFrame:
    """Equal-count bins of the predictions: mean prediction, observed goal rate and a
    match-bootstrap 95% interval of the observed rate."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    order = np.argsort(p, kind="mergesort")
    bin_of = np.empty(len(p), int)
    bin_of[order] = np.arange(len(p)) * n_bins // len(p)
    codes, _ = pd.factorize(pd.Series(match_id))
    rng = np.random.default_rng(seed)
    n_m = codes.max() + 1
    rates = np.empty((B, n_bins))
    for b in range(B):
        w = np.bincount(rng.integers(0, n_m, n_m), minlength=n_m)[codes].astype(np.float64)
        rates[b] = np.bincount(bin_of, w * y, n_bins) / np.maximum(np.bincount(bin_of, w, n_bins), 1e-12)
    n = np.bincount(bin_of, minlength=n_bins)
    out = pd.DataFrame({
        "n": n,
        "mean_pred": np.bincount(bin_of, p, n_bins) / n,
        "goal_rate": np.bincount(bin_of, y, n_bins) / n,
        "ci_low": np.percentile(rates, 2.5, axis=0),
        "ci_high": np.percentile(rates, 97.5, axis=0),
    })
    return out


def expected_calibration_error(bins: pd.DataFrame) -> float:
    return float((bins["n"] * (bins["mean_pred"] - bins["goal_rate"]).abs()).sum() / bins["n"].sum())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def name(c: str) -> str:
    base, tta = (c[:-4], " + mirror TTA") if c.endswith("+tta") else (c, "")
    return C.DISPLAY_NAMES.get(base, base) + tta


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    fmt = lambda v: (format(v, floatfmt) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                     else ("–" if isinstance(v, (float, np.floating)) else str(v)))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main_table(t: pd.DataFrame, cols, group: str) -> pd.DataFrame:
    sub = t[t["group"] == group].set_index("model").loc[[c for c in cols if c in set(t["model"])]]
    return pd.DataFrame({"model": [name(c) for c in sub.index], "log loss": sub["log_loss"].values,
                         "Brier": sub["brier"].values, "ROC-AUC": sub["roc_auc"].values,
                         "mean xG": sub["mean_pred"].values, "goal rate": sub["goal_rate"].values,
                         "shots": sub["n"].astype(int).values})


def full_report(split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from src.visualize import plot_reliability

    pred_path = C.MODELS_DIR / f"predictions_{split}.csv"
    pred = predict_all(split)
    pred.to_csv(pred_path, index=False)
    print(f"predictions -> {pred_path}")

    main_cols = [*C.EVAL_MODELS, *C.BASELINES]
    tta_cols = [f"{r}+tta" for r in C.TTA_MODELS if f"{r}+tta" in pred]
    t = metrics_table(pred, main_cols + tta_cols)
    seeds = seed_table(pred, C.SEED_GROUPS)

    print("bootstrap ...")
    boot_cols = [c for c in main_cols if c != "constant"]
    boot = bootstrap_by_match(pred, boot_cols)
    ci_models, ci_pairs = bootstrap_summary(pred, boot, C.BOOTSTRAP_PAIRS)
    main_pair = C.BOOTSTRAP_PAIRS[0]
    gender_pairs = []
    for g, sub in pred.groupby("gender"):
        sub = sub.reset_index(drop=True)
        _, gp = bootstrap_summary(sub, bootstrap_by_match(sub, list(main_pair)), [main_pair])
        gender_pairs.append(gp.assign(group=g))
    gender_pairs = pd.concat(gender_pairs)

    print("calibration ...")
    cal = {c: reliability_bins(pred[c], pred["is_goal"], pred["match_id"]) for c in C.CALIBRATION_MODELS}
    cal_g = {g: {c: reliability_bins(sub[c], sub["is_goal"], sub["match_id"]) for c in C.CALIBRATION_MODELS}
             for g, sub in pred.groupby("gender")}
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability(cal, pred, C.PLOTS_DIR / f"calibration_{split}.png",
                     title=f"Calibration on the {split} split ({len(pred):,} shots)")
    plot_reliability(cal_g, pred, C.PLOTS_DIR / f"calibration_{split}_by_gender.png", by_group=True,
                     title=f"Calibration by gender ({split} split)")

    # ---- markdown report
    L = [f"# Results on the {split} split", "",
         f"{len(pred):,} open-play shots from {pred['match_id'].nunique():,} matches, goal rate "
         f"{pred['is_goal'].mean():.4f}. CNN rows are the seed-42 runs; the epoch of every CNN was "
         "selected on the validation split. Generated by `python -m src.evaluate --split " + split + "`.", ""]
    for g in ("all", "male", "female"):
        L += [f"## Metrics ({g})", "", md_table(main_table(t, main_cols, g)), ""]
    L += ["## Seeds (mean ± std over seeds)", ""]
    s = seeds.copy()
    s.insert(0, "runs", s.pop("model").map(name))
    for m in METRICS:
        s[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(s.pop(f"{m}_mean"), s.pop(f"{m}_std"))]
    L += [md_table(s), ""]
    L += ["## Test-time augmentation (mean of original and mirrored prediction; not used in the main rows)", "",
          md_table(main_table(t, [c for x in C.TTA_MODELS for c in (x, f"{x}+tta")], "all")), ""]
    L += [f"## Match-level bootstrap, 95% CIs ({C.BOOTSTRAP_B} resamples of matches)", ""]
    cm = ci_models.copy()
    cm["model"] = cm["model"].map(name)
    cm["95% CI"] = [f"[{a:.4f}, {b:.4f}]" for a, b in zip(cm.pop("ci_low"), cm.pop("ci_high"))]
    L += [md_table(cm), ""]
    L += ["### Paired differences a − b (negative log loss / Brier and positive AUC favour a)", ""]
    for d, title in ((ci_pairs, "all shots"), (gender_pairs, "by gender, main pair")):
        d = d.copy()
        d["a"], d["b"] = d["a"].map(name), d["b"].map(name)
        d["95% CI"] = [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(d.pop("ci_low"), d.pop("ci_high"))]
        d["diff"] = d["diff"].map(lambda v: f"{v:+.4f}")
        d["share of resamples where a is better"] = d.pop("p_a_better").map(lambda v: f"{v:.3f}")
        L += [f"**{title}**", "", md_table(d), ""]
    L += ["## Calibration", "", f"Equal-count bins ({C.CALIBRATION_BINS}); ECE = shot-weighted mean "
          "|mean prediction − goal rate| over the bins.", ""]
    ece = pd.DataFrame({"model": [name(c) for c in cal],
                        "ECE": [expected_calibration_error(b) for b in cal.values()],
                        **{f"ECE {g}": [expected_calibration_error(cal_g[g][c]) for c in cal] for g in cal_g}})
    L += [md_table(ece), "", f"![calibration](plots/calibration_{split}.png)", "",
          f"![calibration by gender](plots/calibration_{split}_by_gender.png)", ""]
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / f"metrics_{split}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    t.to_csv(C.RESULTS_DIR / f"metrics_{split}.csv", index=False)
    print(f"report -> {out}")
    print(md_table(main_table(t, main_cols, "all")))


def reference_report():
    from src.dataset import load_shots

    train_rate = load_shots("train")["is_goal"].mean()
    for split in ("val", "test"):
        df = load_shots(split)
        res = {n: compute_metrics(df["is_goal"], p) for n, p in reference_predictions(df, train_rate).items()}
        print(f"\n[{split}] n = {len(df)}, goal rate {df['is_goal'].mean():.4f}")
        print(format_table(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default=None,
                    help="full report on this split; without it only the reference metrics are printed")
    args = ap.parse_args()
    if args.split is None:
        reference_report()
    else:
        full_report(args.split)
