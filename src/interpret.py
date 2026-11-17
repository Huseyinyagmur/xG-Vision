"""Grad-CAM views and the shots where the CNN and StatsBomb xG disagree most.

    python -m src.interpret --compare-layers        # val only: Grad-CAM of several layers, to pick one
    python -m src.interpret --split test            # after `python -m src.evaluate --split test`

Disagreement is measured on the logit scale, logit(p_cnn) - logit(p_statsbomb), so that a
0.02 -> 0.08 change counts as much as 0.2 -> 0.5. Per direction (CNN higher / CNN lower)
the INTERESTING_PER_GROUP goals and non-goals with the largest disagreement are shown.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config as C
from src.dataset import ShotDataset, load_shots
from src.evaluate import md_table, name
from src.gradcam import GradCAM, smallcnn_layer
from src.models import load_checkpoint
from src.utils import get_device
from src.visualize import EXTENT, plot_freeze_frame

STATSBOMB_ONLY = ("technique", "first_time", "one_on_one", "under_pressure")   # inputs our CNN does not get
COMPARE_BLOCKS = (1, 2, 3)                      # 40x30, 20x15, 10x7 feature maps


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gradcams(run: str, df: pd.DataFrame, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM maps (N, H, W) and P(goal) (N,) of `run` for the shots in `df` (no augmentation)."""
    device = get_device()
    model, _ = load_checkpoint(C.MODELS_DIR / run / "best.pt", device)
    cam = GradCAM(model, smallcnn_layer(model, block))
    ds = ShotDataset(df.reset_index(drop=True))
    imgs, extras = zip(*[(ds[i][0], ds[i][1]) for i in range(len(ds))])
    maps, probs = cam(torch.stack(imgs).to(device), torch.stack(extras).to(device))
    cam.remove()
    return maps, probs


def draw_shot(ax, row, cam_map=None, title=None):
    plot_freeze_frame(row, ax)
    if cam_map is not None:
        ax.imshow(np.ma.masked_less(cam_map, 0.15), extent=EXTENT, origin="upper", cmap="inferno",
                  alpha=0.55, vmin=0, vmax=1, zorder=1)
    ax.set_xlim(C.X_MIN + 18, C.X_MAX + 1)        # zoom on the last ~40 yards
    if title is not None:
        ax.set_title(title, fontsize=8.5)


def shot_label(row, p_cnn) -> str:
    flags = [f for f in ("first_time", "one_on_one", "under_pressure") if getattr(row, f)]
    return (f"{'GOAL' if row.is_goal else 'no goal'} | SB {row.statsbomb_xg:.2f} -> CNN {p_cnn:.2f}\n"
            f"{row.body_part}, {row.technique}" + (f", {', '.join(flags)}" if flags else ""))


# ---------------------------------------------------------------------------
def compare_layers(n: int = 6, seed: int = 0) -> None:
    """Val only: the same shots explained at several layers, to choose GRADCAM_LAYER."""
    val = load_shots("val")
    goals = val[val.is_goal == 1].sample(n // 2, random_state=seed)
    misses = val[(val.is_goal == 0) & (val.statsbomb_xg > 0.15)].sample(n - n // 2, random_state=seed)
    shots = pd.concat([goals, misses]).reset_index(drop=True)
    cams = {b: gradcams(C.INTERESTING_MODEL, shots, b) for b in COMPARE_BLOCKS}
    fig, axes = plt.subplots(n, len(COMPARE_BLOCKS), figsize=(4.2 * len(COMPARE_BLOCKS), 3.5 * n))
    for i, row in enumerate(shots.itertuples()):
        for j, b in enumerate(COMPARE_BLOCKS):
            maps, probs = cams[b]
            h, w = smallcnn_shape(b)
            draw_shot(axes[i, j], row, maps[i],
                      f"block {b} ({h}x{w})\n" + shot_label(row, probs[i]) if i == 0 else shot_label(row, probs[i]))
    fig.suptitle(f"Grad-CAM of {name(C.INTERESTING_MODEL)} at different layers (validation shots)", y=1.01)
    fig.tight_layout()
    out = C.PLOTS_DIR / "gradcam_layers_val.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def smallcnn_shape(block: int) -> tuple[int, int]:
    h, w = C.IMG_H, C.IMG_W
    for _ in range(block + 1):
        h, w = h // 2, w // 2
    return h, w


# ---------------------------------------------------------------------------
def select_disagreements(df: pd.DataFrame, p_cnn: np.ndarray, k: int) -> dict[str, pd.DataFrame]:
    d = df.assign(p_cnn=p_cnn, dlogit=logit(p_cnn) - logit(df["statsbomb_xg"]))
    out = {}
    for direction, asc in (("cnn_higher", False), ("cnn_lower", True)):
        parts = [d[d.is_goal == g].sort_values("dlogit", ascending=asc).head(k) for g in (1, 0)]
        out[direction] = pd.concat(parts)
    return out


def statsbomb_only_table(df: pd.DataFrame, p_cnn: np.ndarray, top_frac: float = 0.05) -> pd.DataFrame:
    """How often StatsBomb-only attributes occur among the strongest disagreements vs all shots."""
    d = df.assign(dlogit=logit(p_cnn) - logit(df["statsbomb_xg"]))
    k = int(len(d) * top_frac)
    groups = {"all shots": d,
              f"top {top_frac:.0%} CNN higher": d.nlargest(k, "dlogit"),
              f"top {top_frac:.0%} CNN lower": d.nsmallest(k, "dlogit")}
    rows = {}
    for gname, g in groups.items():
        r = {"shots": len(g), "goal rate": g.is_goal.mean(), "mean SB xG": g.statsbomb_xg.mean()}
        for f in ("first_time", "one_on_one", "under_pressure"):
            r[f] = g[f].mean()
        r["header"] = (g.body_part == "Head").mean()
        r["technique not 'Normal'"] = (g.technique != "Normal").mean()
        for t in ("Volley", "Half Volley", "Lob"):
            r[t] = (g.technique == t).mean()
        rows[gname] = r
    t = pd.DataFrame(rows).T.reset_index(names="group")
    t["shots"] = t["shots"].astype(int)
    return t


def interesting_shots(split: str) -> None:
    if C.GRADCAM_LAYER is None:
        raise SystemExit("set GRADCAM_LAYER in src/config.py first (choose it on val with --compare-layers)")
    df = load_shots(split)
    pred = pd.read_csv(C.MODELS_DIR / f"predictions_{split}.csv")
    assert (pred["shot_id"].to_numpy() == df["shot_id"].to_numpy()).all()
    p_cnn = pred[C.INTERESTING_MODEL].to_numpy()
    sel = select_disagreements(df, p_cnn, C.INTERESTING_PER_GROUP)

    titles = {"cnn_higher": f"{name(C.INTERESTING_MODEL)} much HIGHER than StatsBomb xG",
              "cnn_lower": f"{name(C.INTERESTING_MODEL)} much LOWER than StatsBomb xG"}
    k = C.INTERESTING_PER_GROUP
    rows_md = []
    for direction, shots in sel.items():
        maps, probs = gradcams(C.INTERESTING_MODEL, shots, C.GRADCAM_LAYER)
        assert np.allclose(probs, shots["p_cnn"], atol=2e-2), "Grad-CAM forward differs from the predictions"
        fig, axes = plt.subplots(2, k, figsize=(4.6 * k, 8.2))
        for ax, row, m in zip(axes.flat, shots.itertuples(), maps):
            draw_shot(ax, row, m, shot_label(row, row.p_cnn))
        axes[0, 0].legend(loc="lower left", fontsize=6.5)
        fig.suptitle(f"{titles[direction]} ({split} split; top row goals, bottom row no goal)\n"
                     f"overlay: Grad-CAM, block {C.GRADCAM_LAYER}", fontsize=11)
        fig.tight_layout()
        out = C.PLOTS_DIR / f"interesting_{direction}_{split}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")
        for row in shots.itertuples():
            rows_md.append({"direction": direction, "outcome": "goal" if row.is_goal else "no goal",
                            "SB xG": row.statsbomb_xg, "CNN xG": row.p_cnn, "distance (yd)": row.distance,
                            "body part": row.body_part, "technique": row.technique,
                            "first time": row.first_time, "one on one": row.one_on_one,
                            "under pressure": row.under_pressure, "competition": row.competition,
                            "shot_id": row.shot_id})
    L = [f"# Shots where {name(C.INTERESTING_MODEL)} and StatsBomb xG disagree most ({split} split)", "",
         "Selection: largest |logit(p_CNN) − logit(p_StatsBomb)| per direction, "
         f"{k} goals and {k} non-goals each. Grad-CAM layer: SmallCNN block {C.GRADCAM_LAYER}.", "",
         md_table(pd.DataFrame(rows_md), floatfmt=".3f"), "",
         "## StatsBomb-only attributes among the strongest disagreements", "",
         "Share of shots with each attribute. These inputs are used by StatsBomb's model but not by the CNN.", "",
         md_table(statsbomb_only_table(df, p_cnn), floatfmt=".3f"), "",
         f"![CNN higher](plots/interesting_cnn_higher_{split}.png)", "",
         f"![CNN lower](plots/interesting_cnn_lower_{split}.png)", ""]
    out = C.RESULTS_DIR / f"interesting_{split}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-layers", action="store_true", help="val only: Grad-CAM at several layers")
    ap.add_argument("--split", choices=["val", "test"], default=None)
    args = ap.parse_args()
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.compare_layers:
        compare_layers()
    if args.split:
        interesting_shots(args.split)
