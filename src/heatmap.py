"""xG heat maps: the shooter is moved over a grid while the rest of the situation stays fixed.

    python -m src.heatmap

Scenarios
    empty    : only the goalkeeper, on the goal line at the centre ("[OOD]": no outfield player
               at all never happens in training)
    crowded  : a real test freeze frame with >= 6 outfield defenders in the penalty box
    counter  : a real test freeze frame with <= 2 outfield defenders within 30 yd of goal
The two real frames are chosen by rule with a fixed seed (not by hand). Foot and head are
mapped separately; headers from 25+ yd are out of distribution (dashed contour).

The logistic regression (distance + angle + body part) only sees the shooter position, so its
map is the same in every scenario. Differences:
    CNN(empty)  - LogReg    : what the CNN learned about geometry beyond distance and angle
    CNN(real)   - CNN(empty): how the CNN accounts for the defenders and teammates
Outputs: results/heatmaps.md, results/plots/heatmap_{empty,crowded,counter}.png
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

from src import config as C
from src import perturb as P
from src.baselines import fit_logreg
from src.dataset import load_shots
from src.models import load_checkpoint
from src.sanity import ShotList, logreg_body, predict_logits, sigmoid
from src.utils import get_device

BOX = dict(x=102.0, y_lo=18.0, y_hi=62.0)        # penalty area
REF_POINTS = {                                    # (x, y) for the summary table
    "6-yd box centre (114, 40)": (114, 40), "penalty spot (108, 40)": (108, 40),
    "edge of box, centre (102, 40)": (102, 40), "edge of box, wide (102, 25)": (102, 25),
    "tight angle (116, 30)": (116, 30), "25 yd, centre (95, 40)": (95, 40),
}
DIFF_LIM = 0.25


def grid() -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(C.HEATMAP_X[0], C.HEATMAP_X[1] + 1e-9, C.HEATMAP_STEP)
    ys = np.arange(C.HEATMAP_Y[0], C.HEATMAP_Y[1] + 1e-9, C.HEATMAP_STEP)
    return xs, ys


def pick_frames(df: pd.DataFrame) -> dict[str, tuple[P.Shot, str]]:
    """Rule-based real situations (fixed seed): one crowded, one counter-attack frame."""
    rng = np.random.default_rng(C.SANITY_EXAMPLE_SEED)
    crowded, counter = [], []
    for i, r in enumerate(df.itertuples()):
        s = P.Shot.from_row(r)
        k = s.gk_index()
        if k is None or s.ff[k, 0] < 114:                 # keeper present and near the goal
            continue
        opp = s.ff[(s.ff[:, 2] == 0) & (s.ff[:, 3] == 0)]
        in_box = ((opp[:, 0] >= BOX["x"]) & (opp[:, 1] >= BOX["y_lo"]) & (opp[:, 1] <= BOX["y_hi"])).sum()
        near = (np.hypot(C.GOAL_X - opp[:, 0], C.GOAL_CENTER[1] - opp[:, 1]) <= 30).sum()
        if in_box >= 6:
            crowded.append(i)
        if near <= 2:
            counter.append(i)
    out = {}
    for name, idx in (("crowded", crowded), ("counter", counter)):
        row = df.iloc[int(rng.choice(idx))]
        out[name] = (P.Shot.from_row(row), row.shot_id)
    return out


def predict_maps(model, device, scenarios: dict[str, np.ndarray], lr) -> dict:
    """{(scenario, body): (len(ys), len(xs)) array of P(goal)}; scenario 'logreg' for the baseline."""
    xs, ys = grid()
    X, Y = np.meshgrid(xs, ys)
    maps = {}
    for body in ("foot", "head"):
        shots_lr = [P.Shot(float(x), float(y), P.keeper_only(), body) for x, y in zip(X.ravel(), Y.ravel())]
        maps[("logreg", body)] = logreg_body(shots_lr, lr).reshape(X.shape)
        for name, ff in scenarios.items():
            shots = [P.Shot(float(x), float(y), ff, body) for x, y in zip(X.ravel(), Y.ravel())]
            maps[(name, body)] = sigmoid(predict_logits(model, ShotList(shots), device)).reshape(X.shape)
    return maps


# ---------------------------------------------------------------------------
def _draw_players(pitch, ax, ff):
    team, gk = ff[:, 2] == 1, ff[:, 3] == 1
    pitch.scatter(ff[team, 0], ff[team, 1], ax=ax, c="tab:blue", s=30, ec="k", zorder=4)
    pitch.scatter(ff[~team & ~gk, 0], ff[~team & ~gk, 1], ax=ax, c="tab:red", s=30, ec="k", zorder=4)
    pitch.scatter(ff[~team & gk, 0], ff[~team & gk, 1], ax=ax, c="black", s=40, marker="s", zorder=4)


def _panel(ax, z, kind, title, ff=None, body="foot"):
    from matplotlib.colors import TwoSlopeNorm
    from mplsoccer import Pitch
    xs, ys = grid()
    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="#444", pitch_color="white", line_zorder=3)
    pitch.draw(ax=ax)
    ext = [xs[0] - 0.5, xs[-1] + 0.5, ys[-1] + 0.5, ys[0] - 0.5]
    if kind == "xg":
        im = ax.imshow(z, extent=ext, origin="upper", cmap="viridis", vmin=0, vmax=0.6, zorder=1)
        cs = ax.contour(xs, ys, z, levels=[0.05, 0.1, 0.2, 0.4], colors="white", linewidths=0.7, zorder=2)
        ax.clabel(cs, fmt="%.2f", fontsize=6)
    else:
        im = ax.imshow(z, extent=ext, origin="upper", cmap="RdBu_r",
                       norm=TwoSlopeNorm(0, -DIFF_LIM, DIFF_LIM), zorder=1)
    if body == "head":                                  # headers from 25+ yd: OOD
        d, _ = P.distance_angle(*np.meshgrid(xs, ys))
        ax.contour(xs, ys, d, levels=[P.OOD_HEADER_DISTANCE], colors="k", linestyles="--", linewidths=1, zorder=2)
    if ff is not None:
        _draw_players(pitch, ax, ff)
    ax.set_xlim(C.HEATMAP_X[0] - 1, C.X_MAX + 1)
    ax.set_title(title, fontsize=9)
    return im


def plot_scenario(maps, name, ff, path, empty: bool):
    import matplotlib.pyplot as plt
    if empty:
        cols = [(lambda b: maps[("empty", b)], "xg", "CNN, keeper only [OOD]"),
                (lambda b: maps[("logreg", b)], "xg", "LogReg (distance + angle + body part)"),
                (lambda b: maps[("empty", b)] - maps[("logreg", b)], "diff", "CNN − LogReg")]
    else:
        cols = [(lambda b: maps[(name, b)], "xg", f"CNN, real {name} frame"),
                (lambda b: maps[(name, b)] - maps[("empty", b)], "diff", "CNN(real) − CNN(keeper only)\n= effect of the other players"),
                (lambda b: maps[(name, b)] - maps[("logreg", b)], "diff", "CNN(real) − LogReg")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6))
    for r, body in enumerate(("foot", "head")):
        for c, (fn, kind, title) in enumerate(cols):
            im = _panel(axes[r, c], fn(body), kind, f"{title} — {body}", None if empty else ff, body)
            fig.colorbar(im, ax=axes[r, c], fraction=0.035, pad=0.02,
                         label="xG" if kind == "xg" else "Δ xG")
    sup = ("xG over a shooter grid, keeper alone on the goal line" if empty else
           f"xG over a shooter grid, defence and teammates of a real '{name}' test frame fixed")
    fig.suptitle(sup + "\n(dashed: 25 yd, headers beyond are out of distribution)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def asymmetry_table(maps) -> pd.DataFrame:
    """Keeper-only map vs. its mirror about y = 40 (the grid is symmetric about 40)."""
    xs, ys = grid()
    assert np.allclose(ys + ys[::-1], 2 * C.GOAL_CENTER[1])
    recs = []
    for key in (("empty", "foot"), ("empty", "head"), ("logreg", "foot")):
        z = maps[key]
        diff = np.abs(z - z[::-1])
        k = np.unravel_index(diff.argmax(), diff.shape)
        name = "CNN" if key[0] == "empty" else "LogReg"
        recs.append({"map": f"{name}, {key[1]}", "mean |p(y) − p(80−y)|": diff.mean(),
                     "max": diff.max(), "at (x, y)": f"({xs[k[1]]:.0f}, {ys[k[0]]:.0f})",
                     "p there": z[k], "p mirrored": z[::-1][k]})
    return pd.DataFrame(recs)


def monotonic_table(maps) -> pd.DataFrame:
    """Keeper-only map: in the corridor between the posts, moving 1 yd towards the goal line
    gets closer and widens the angle, so xG should not fall."""
    xs, ys = grid()
    corridor = (ys > C.POST_LOW_Y) & (ys < C.POST_HIGH_Y)
    recs = []
    for key in (("empty", "foot"), ("empty", "head"), ("logreg", "foot")):
        z = maps[key][corridor]
        step = np.diff(z, axis=1)                     # x increases along axis 1
        recs.append({"map": f"{'CNN' if key[0] == 'empty' else 'LogReg'}, {key[1]}", "steps": step.size,
                     f"share falling > {C.SANITY_UNCHANGED:g}": float((step < -C.SANITY_UNCHANGED).mean()),
                     "share falling > 0.01": float((step < -0.01).mean()),
                     "falls > 0.01 at x >= 117": int(((step < -0.01) & (xs[1:] >= 117)[None, :]).sum()),
                     "largest fall": float(-step.min()) if step.min() < 0 else 0.0})
    return pd.DataFrame(recs)


def goal_line_table(maps, train: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Keeper-only map along the last grid columns, wide of the posts, plus how many training
    shots were taken from there."""
    xs, ys = grid()
    recs = []
    for y in (12.0, 20.0, 28.0):
        i = int(np.argmin(np.abs(ys - y)))
        r = {"y (wide of the posts)": f"{y:g}"}
        for x in (115.0, 117.0, 118.0, 119.0):
            r[f"CNN x={x:g}"] = maps[("empty", "foot")][i, int(np.argmin(np.abs(xs - x)))]
        r["LogReg x=119"] = maps[("logreg", "foot")][i, int(np.argmin(np.abs(xs - 119)))]
        recs.append(r)
    near = train[(train.x >= 118.5) & ((train.y - C.GOAL_CENTER[1]).abs() >= 8)]
    note = (f"Training shots with x >= 118.5 and |y − 40| >= 8: {len(near)} of {len(train):,} "
            f"(goal rate {near.is_goal.mean():.2f}).")
    return pd.DataFrame(recs), note


def reference_table(maps, scenarios) -> pd.DataFrame:
    xs, ys = grid()
    recs = []
    for label, (x, y) in REF_POINTS.items():
        j, i = int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))
        r = {"point": label}
        for body in ("foot", "head"):
            r[f"LogReg {body}"] = maps[("logreg", body)][i, j]
            for name in scenarios:
                r[f"CNN {name} {body}"] = maps[(name, body)][i, j]
        recs.append(r)
    return pd.DataFrame(recs)


def defence_table(maps, scenarios) -> pd.DataFrame:
    """Effect of the real players (CNN(real) − CNN(keeper only)) inside the penalty box."""
    xs, ys = grid()
    X, Y = np.meshgrid(xs, ys)
    box = (X >= BOX["x"]) & (Y >= BOX["y_lo"]) & (Y <= BOX["y_hi"])
    recs = []
    for name in scenarios:
        if name == "empty":
            continue
        for body in ("foot", "head"):
            d = (maps[(name, body)] - maps[("empty", body)])[box]
            k = np.argmin(d)
            recs.append({"frame": name, "body": body, "mean Δ in box": d.mean(),
                         "share of box points lower": float((d < -C.SANITY_UNCHANGED).mean()),
                         "share higher": float((d > C.SANITY_UNCHANGED).mean()),
                         "largest drop": d.min(), "at (x, y)": f"({X[box][k]:.0f}, {Y[box][k]:.0f})",
                         "largest rise": d.max()})
    return pd.DataFrame(recs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluate import md_table

    device = get_device()
    test = load_shots("test").reset_index(drop=True)
    frames = pick_frames(test)
    scenarios = {"empty": P.keeper_only(), **{k: s.ff for k, (s, _) in frames.items()}}
    model, _ = load_checkpoint(C.MODELS_DIR / C.SANITY_MODEL / "best.pt", device)
    train = load_shots("train")
    lr = fit_logreg(train, body_part=True)
    maps = predict_maps(model, device, scenarios, lr)

    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_scenario(maps, "empty", None, C.PLOTS_DIR / "heatmap_empty.png", empty=True)
    for name, (s, _) in frames.items():
        plot_scenario(maps, name, s.ff, C.PLOTS_DIR / f"heatmap_{name}.png", empty=False)

    xs, ys = grid()
    gl_table, gl_note = goal_line_table(maps, train)
    info = pd.DataFrame([{"frame": k, "shot_id": sid,
                          "outfield defenders": int(((s.ff[:, 2] == 0) & (s.ff[:, 3] == 0)).sum()),
                          "teammates": int((s.ff[:, 2] == 1).sum())} for k, (s, sid) in frames.items()])
    L = ["# xG heat maps", "",
         f"Model `{C.SANITY_MODEL}` (seed 42). Shooter grid x ∈ [{xs[0]:.0f}, {xs[-1]:.0f}], "
         f"y ∈ [{ys[0]:.0f}, {ys[-1]:.0f}], {C.HEATMAP_STEP:g} yd steps ({xs.size} × {ys.size} points). "
         "The keeper-only scenario is out of distribution (no outfield players never occurs in training); "
         "headers from 25+ yd are out of distribution too. Generated by `python -m src.heatmap`.", "",
         "Real frames (chosen by rule from the test split, fixed seed):", "", md_table(info), "",
         "## xG at reference points", "", md_table(reference_table(maps, scenarios), floatfmt=".3f"), "",
         "## Mirror asymmetry of the keeper-only map (about y = 40)", "",
         md_table(asymmetry_table(maps), floatfmt=".4f"), "",
         "## Monotonicity between the posts (keeper-only map)", "",
         "Between the posts, every 1 yd step towards the goal line is closer and wider; xG should not fall.", "",
         md_table(monotonic_table(maps), floatfmt=".4f"), "",
         "## Goal-line edge (keeper-only map, foot)", "",
         "The last grid column (x = 119) lies 1 yd from the goal line, at the right edge of the tensor.", "",
         md_table(gl_table, floatfmt=".3f"), "", gl_note, "",
         "## Effect of the real players inside the penalty box", "",
         md_table(defence_table(maps, scenarios), floatfmt=".3f"), "",
         "![empty](plots/heatmap_empty.png)", "", "![crowded](plots/heatmap_crowded.png)", "",
         "![counter](plots/heatmap_counter.png)", ""]
    out = C.RESULTS_DIR / "heatmaps.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:-6]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
