"""Plotting helpers: freeze frames on a pitch and the corresponding tensor channels."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch, Pitch

from src import config as C
from src.pitch_tensor import render_shot

# StatsBomb/mplsoccer convention: y grows downwards -> row 0 (y=0) at the top
EXTENT = [C.X_MIN, C.X_MAX, C.Y_MAX, C.Y_MIN]


def plot_freeze_frame(row, ax=None):
    """Classic scatter view of one shot on the attacking half."""
    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="#888", pitch_color="white")
    if ax is None:
        fig, ax = pitch.draw(figsize=(6, 5))
    else:
        pitch.draw(ax=ax)
    ff = row.freeze_frame
    team, gk = ff[:, 2] == 1, ff[:, 3] == 1
    pitch.polygon([[(row.x, row.y), (C.GOAL_X, C.POST_LOW_Y), (C.GOAL_X, C.POST_HIGH_Y)]],
                  ax=ax, color="gold", alpha=0.3)
    pitch.scatter(ff[team, 0], ff[team, 1], ax=ax, c="tab:blue", s=60, ec="k", label="teammate")
    pitch.scatter(ff[~team & ~gk, 0], ff[~team & ~gk, 1], ax=ax, c="tab:red", s=60, ec="k", label="opponent")
    pitch.scatter(ff[~team & gk, 0], ff[~team & gk, 1], ax=ax, c="black", s=80, marker="s", label="GK")
    pitch.scatter(row.x, row.y, ax=ax, c="gold", s=180, marker="*", ec="k", zorder=5, label="shooter")
    ax.set_title(f"{row.player.split()[0]} {row.minute}' | {row.outcome} | SB xG {row.statsbomb_xg:.2f}",
                 fontsize=9)
    return ax


def plot_channels(row, save_path=None):
    """Freeze-frame view + every tensor channel for a single shot."""
    t = render_shot(row.x, row.y, row.freeze_frame)
    fig, axes = plt.subplots(1, C.N_CHANNELS + 1, figsize=(3.2 * (C.N_CHANNELS + 1), 3.6))
    plot_freeze_frame(row, axes[0])
    for c, ax in enumerate(axes[1:]):
        ax.imshow(t[c], origin="upper", extent=EXTENT, cmap="magma", vmin=0, vmax=max(1, t[c].max()))
        ax.set_title(f"{c}: {C.CHANNELS[c]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_examples_grid(df, n=6, seed=0, save_path=None):
    """Grid of freeze frames, half goals / half non-goals."""
    rng = np.random.default_rng(seed)
    goals = df[df.is_goal == 1].sample(n // 2, random_state=seed)
    misses = df[df.is_goal == 0].sample(n - n // 2, random_state=seed)
    rows = list(goals.itertuples()) + list(misses.itertuples())
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(5 * cols, 4.3 * np.ceil(n / cols)))
    for ax, r in zip(axes.flat, rows):
        plot_freeze_frame(r, ax)
    axes.flat[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_data_overview(df, save_path=None):
    """Goal rate vs distance + shot location density (sanity check of the data)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    bins = np.arange(0, 42, 2)
    centers = (bins[:-1] + bins[1:]) / 2
    d = pd.cut(df["distance"], bins, labels=centers)
    agg = df.groupby(d, observed=True)[["is_goal", "statsbomb_xg"]].mean()
    axes[0].plot(agg.index.astype(float), agg["is_goal"], "o-", label="observed goal rate")
    axes[0].plot(agg.index.astype(float), agg["statsbomb_xg"], "s--", label="mean StatsBomb xG")
    axes[0].set_xlabel("distance to goal centre (yd)"); axes[0].set_ylabel("P(goal)")
    axes[0].set_xlim(0, 40); axes[0].legend(); axes[0].set_title("Goal rate vs distance")

    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="#888")
    pitch.draw(ax=axes[1])
    pitch.hexbin(df.x, df.y, ax=axes[1], gridsize=(20, 14), cmap="Blues", mincnt=1, edgecolors="none")
    axes[1].set_title(f"Shot locations (n={len(df)})")

    pitch.draw(ax=axes[2])
    g = df[df.is_goal == 1]
    pitch.scatter(g.x, g.y, ax=axes[2], s=6, alpha=0.4, c="tab:red")
    axes[2].set_title(f"Goals (n={len(g)}, rate={df.is_goal.mean():.3f})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # python -m src.visualize  -> writes the step-1 plots to results/plots/
    import matplotlib
    matplotlib.use("Agg")
    from src.dataset import load_shots

    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_shots()
    plot_data_overview(df, C.PLOTS_DIR / "data_overview.png")
    plot_examples_grid(df[df.split == "train"], n=6, seed=3, save_path=C.PLOTS_DIR / "freeze_frame_examples.png")
    goals = df[df.is_goal == 1].sort_values("statsbomb_xg")
    plot_channels(goals.iloc[len(goals) // 2], C.PLOTS_DIR / "example_channels.png")
    print(f"plots saved to {C.PLOTS_DIR}")
