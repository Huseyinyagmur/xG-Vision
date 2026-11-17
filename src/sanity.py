"""Quantitative sanity tests of the xG CNN (step 4).

    python -m src.sanity --split test

Every shot of the split is edited in a controlled way (src/perturb.py), re-rendered and
re-scored by the SmallCNN (seed 42). For each edit we report on how many shots it was
applicable, the share where xG moved in the expected direction, and the mean / median change,
overall and per distance band. Changes smaller than SANITY_UNCHANGED count as "no change".

Rows whose situation is out of the training distribution (no goalkeeper, headers from 25+ yd)
are flagged OOD and reported separately, so that they are not mixed with real inconsistencies.

Outputs: results/sanity_<split>.md|csv, models/sanity_<split>_rows.csv (all edited rows),
results/plots/sanity_{effects,examples,failures}_<split>.png.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")       # see src/train.py: one BLAS thread per DataLoader worker

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import config as C
from src import perturb as P
from src.baselines import fit_logreg
from src.dataset import load_shots
from src.models import load_checkpoint
from src.utils import get_device, is_memory_error, pick_num_workers

# edit name -> function(Shot) -> Shot (raises P.NotApplicable when it does not apply)
EDITS = {
    "orig": lambda s: s,
    "1a_closer": lambda s: P.move_shooter(s, C.SANITY_STEP),
    "1b_farther": lambda s: P.move_shooter(s, -C.SANITY_STEP),
    "2a_def_near": lambda s: P.add_defender_on_line(s, C.SANITY_DEF_NEAR),
    "2b_def_mid": lambda s: P.add_defender_on_line(s, s.distance / 2),
    "2c_def_aside": lambda s: P.add_defender_aside(s, s.distance / 2, C.SANITY_ASIDE),
    "3a_no_gk": P.remove_goalkeeper,
    "3b_gk_side": lambda s: P.shift_goalkeeper_sideways(s, C.SANITY_STEP),
    "3c_gk_forward": lambda s: P.shift_goalkeeper_forward(s, C.SANITY_STEP),
    "4a_foot_to_head": lambda s: _swap_body(s, "foot", "head"),
    "4b_head_to_foot": lambda s: _swap_body(s, "head", "foot"),
    "5_mirror": P.mirror,
}

# test -> (description, expected sign of the change in xG; None = descriptive / own criterion, 0 = no change)
TESTS = {
    "1a_closer": ("Shooter 3 yd closer to the goal centre", +1),
    "1b_farther": ("Shooter 3 yd farther from the goal centre", -1),
    "2a_def_near": ("Defender added 1.5 yd in front of the shooter, on the shot line", -1),
    "2b_def_mid": ("Defender added halfway along the shot line", -1),
    "2c_def_aside": ("Control: the halfway defender moved 6 yd sideways, outside the cone", None),
    "3a_no_gk": ("Goalkeeper removed", +1),
    "3b_gk_side": ("Goalkeeper moved 3 yd sideways, away from the shot line", +1),
    "3c_gk_forward": ("Goalkeeper moved 3 yd towards the shooter (descriptive)", None),
    "4a_foot_to_head": ("Body part foot -> head (same position)", -1),
    "4b_head_to_foot": ("Body part head -> foot (same position)", +1),
    "5_mirror": ("Mirror image (y -> 80 - y)", 0),
}
DIRECTIONAL = [t for t, (_, e) in TESTS.items() if e in (+1, -1)]
LOGREG_TESTS = ("1a_closer", "1b_farther", "4a_foot_to_head", "4b_head_to_foot")   # LogReg sees these inputs


def _swap_body(s: P.Shot, src: str, dst: str) -> P.Shot:
    if s.body != src:
        raise P.NotApplicable(f"not a {src} shot")
    return P.set_body_part(s, dst)


def band_labels() -> list[str]:
    b = C.DISTANCE_BANDS
    return [f"{lo}-{hi}" if hi < 100 else f"{lo}+" for lo, hi in zip(b[:-1], b[1:])]


def band_of(distance) -> pd.Categorical:
    return pd.cut(distance, C.DISTANCE_BANDS, labels=band_labels(), right=False)


def _scored_rows(rows: pd.DataFrame, test: str) -> tuple[pd.DataFrame, str]:
    """In-distribution rows of a test; if a test only produces OOD rows (goalkeeper removed),
    all its rows are used and the label is marked [OOD]."""
    d = rows[rows.edit == test]
    ind = d[~d.is_ood]
    return (ind, test) if len(ind) else (d, f"{test} [OOD]")


# ---------------------------------------------------------------------------
# Rendering + prediction
# ---------------------------------------------------------------------------
class EditedShots(Dataset):
    """(base shot index, edit name) -> (fp16 image, extra). The edit is applied in the worker."""

    def __init__(self, base: list[P.Shot], items: list[tuple[int, str]]):
        self.base, self.items = base, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, e = self.items[k]
        s = EDITS[e](self.base[i])
        return torch.from_numpy(s.image()).to(torch.float16), torch.from_numpy(s.extra())


class ShotList(Dataset):
    def __init__(self, shots: list[P.Shot]):
        self.shots = shots

    def __len__(self):
        return len(self.shots)

    def __getitem__(self, k):
        s = self.shots[k]
        return torch.from_numpy(s.image()).to(torch.float16), torch.from_numpy(s.extra())


@torch.no_grad()
def _predict_once(model, ds: Dataset, device, batch_size: int, num_workers: int) -> np.ndarray:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    out = []
    for x, e in dl:
        out.append(model(x.to(device, non_blocking=True).float(), e.to(device)).float().reshape(-1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def predict_logits(model, ds: Dataset, device, batch_size: int = 512, num_workers: int | None = None):
    """Logits in float32 (no autocast: the differences we look at can be small).
    Workers: NUM_WORKERS, or 2 when less than 6 GB is free. On an out-of-memory failure the
    worker count and batch size are halved and the whole pass is retried (down to 0 workers)."""
    if num_workers is None:
        num_workers = pick_num_workers(C.NUM_WORKERS)
    while True:
        try:
            return _predict_once(model, ds, device, batch_size, num_workers)
        except (RuntimeError, OSError, MemoryError) as err:
            if not is_memory_error(err) or (num_workers == 0 and batch_size <= 64):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            num_workers, batch_size = num_workers // 2, max(64, batch_size // 2)
            print(f"memory error ({type(err).__name__}: {str(err)[:120]}); "
                  f"retrying with num_workers = {num_workers}, batch_size = {batch_size}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logreg_body(shots: list[P.Shot], model) -> np.ndarray:
    """P(goal) of LogReg (distance + angle + body part) for arbitrary edited shots."""
    d, a = P.distance_angle([s.x for s in shots], [s.y for s in shots])
    body = np.stack([s.extra() for s in shots]).astype(np.float64)
    return model.predict_proba(np.column_stack([d, a, body]))[:, 1]


# ---------------------------------------------------------------------------
def build_rows(df: pd.DataFrame) -> tuple[list[P.Shot], pd.DataFrame, list[P.Shot]]:
    """Apply every edit to every shot; returns base shots, row metadata, edited shots."""
    base = [P.Shot.from_row(r) for r in df.itertuples()]
    meta, edited = [], []
    for i, s in enumerate(base):
        for e, fn in EDITS.items():
            try:
                t = fn(s)
            except P.NotApplicable:
                continue
            ood = sorted(set(s.ood_reasons()) | set(t.ood_reasons()))
            near = np.hypot(t.ff[:, 0] - t.x, t.ff[:, 1] - t.y).min() if len(t.ff) else np.inf
            meta.append({"i": i, "edit": e, "ood": "; ".join(ood), "distance_after": t.distance,
                         "player_within_1yd": bool(near < 1.0)})
            edited.append(t)
    return base, pd.DataFrame(meta), edited


def run(split: str):
    t0 = time.time()
    device = get_device()
    df = load_shots(split).reset_index(drop=True)
    d_chk, a_chk = P.distance_angle(df.x, df.y)
    assert np.allclose(d_chk, df.distance) and np.allclose(a_chk, df.angle), "geometry differs from the data"

    base, rows, edited = build_rows(df)
    print(f"{len(df):,} shots -> {len(rows):,} rendered situations ({time.time() - t0:.0f} s)")

    model, _ = load_checkpoint(C.MODELS_DIR / C.SANITY_MODEL / "best.pt", device)
    ds = EditedShots(base, list(zip(rows["i"], rows["edit"])))
    rows["logit"] = predict_logits(model, ds, device)
    rows["p"] = sigmoid(rows["logit"])
    print(f"predicted ({time.time() - t0:.0f} s)")

    lr = fit_logreg(load_shots("train"), body_part=True)
    rows["p_logreg"] = logreg_body(edited, lr)

    orig = rows[rows.edit == "orig"].set_index("i")
    rows = rows[rows.edit != "orig"].copy()
    for c in ("p", "logit", "p_logreg"):
        rows[f"{c}_orig"] = rows["i"].map(orig[c])
    rows["dp"] = rows["p"] - rows["p_orig"]
    rows["dlogit"] = rows["logit"] - rows["logit_orig"]
    rows["dp_logreg"] = rows["p_logreg"] - rows["p_logreg_orig"]
    info = df[["shot_id", "distance", "body_part", "is_goal", "gender", "statsbomb_xg"]]
    rows = rows.join(info, on="i")
    rows["band"] = band_of(rows["distance"])
    rows["is_ood"] = rows["ood"] != ""

    out_rows = C.MODELS_DIR / f"sanity_{split}_rows.csv"
    rows.to_csv(out_rows, index=False)
    print(f"-> {out_rows}")
    report(split, df, base, rows, orig)
    print(f"done in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def direction_shares(dp: pd.Series, expected: int) -> dict:
    moved = dp.abs() >= C.SANITY_UNCHANGED
    return {"expected": float(((np.sign(dp) == expected) & moved).mean()),
            "unchanged": float((~moved).mean()),
            "opposite": float(((np.sign(dp) == -expected) & moved).mean())}


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, (desc, exp) in TESTS.items():
        d = rows[rows.edit == t]
        ind, label = _scored_rows(rows, t)
        r = {"test": label, "description": desc, "shots": len(d), "OOD": int(d.is_ood.sum())}
        if exp in (+1, -1):
            sh = direction_shares(ind.dp, exp)
            r.update({"expected direction": sh["expected"], "no change": sh["unchanged"],
                      "opposite": sh["opposite"]})
        r.update({"mean Δp": ind.dp.mean(), "median Δp": ind.dp.median(), "mean Δlogit": ind.dlogit.mean()})
        out.append(r)
    return pd.DataFrame(out)


def control_table(rows: pd.DataFrame) -> pd.DataFrame:
    """2c criterion: per shot, |Δ2c| < |Δ2a| and |Δ2c| < |Δ2b| (shots where all three apply)."""
    w = rows[rows.edit.isin(["2a_def_near", "2b_def_mid", "2c_def_aside"])]
    piv = w.pivot_table(index="i", columns="edit", values="dp")
    flags = w.groupby("i").is_ood.any()
    piv = piv.dropna()
    piv = piv[~flags.reindex(piv.index).fillna(False).to_numpy()]
    a, b, c = (piv[k].abs() for k in ("2a_def_near", "2b_def_mid", "2c_def_aside"))
    return pd.DataFrame([{
        "shots": len(piv),
        "|Δ2c| < |Δ2a|": float((c < a).mean()), "|Δ2c| < |Δ2b|": float((c < b).mean()),
        "|Δ2c| < both (criterion)": float(((c < a) & (c < b)).mean()),
        "mean Δ2c": piv["2c_def_aside"].mean(), "share Δ2c < 0": float((piv["2c_def_aside"] < 0).mean()),
    }])


def descriptive_table(rows: pd.DataFrame, test: str) -> pd.DataFrame:
    d = rows[(rows.edit == test) & ~rows.is_ood]
    g = d.groupby("band", observed=True)
    t = pd.DataFrame({"shots": g.size(), "mean Δp": g.dp.mean(), "median Δp": g.dp.median(),
                      "share up": g.dp.apply(lambda x: (x >= C.SANITY_UNCHANGED).mean()),
                      "share down": g.dp.apply(lambda x: (x <= -C.SANITY_UNCHANGED).mean())})
    return t.reset_index()


def band_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Share in the expected direction per distance band (in-distribution rows)."""
    recs = []
    for t in DIRECTIONAL:
        exp = TESTS[t][1]
        d, label = _scored_rows(rows, t)
        for band, g in d.groupby("band", observed=True):
            recs.append({"test": label, "band": band, "shots": len(g),
                         "expected": direction_shares(g.dp, exp)["expected"], "mean Δp": g.dp.mean()})
    return pd.DataFrame(recs)


def logreg_table(rows: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for t in LOGREG_TESTS:
        exp = TESTS[t][1]
        d = rows[(rows.edit == t) & ~rows.is_ood]
        recs.append({"test": t, "shots": len(d),
                     "CNN expected": direction_shares(d.dp, exp)["expected"], "CNN mean Δp": d.dp.mean(),
                     "LogReg expected": direction_shares(d.dp_logreg, exp)["expected"],
                     "LogReg mean Δp": d.dp_logreg.mean()})
    return pd.DataFrame(recs)


def mirror_table(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows[rows.edit == "5_mirror"]
    a = d.dp.abs()
    return pd.DataFrame([{"shots": len(d), "mean |Δp|": a.mean(), "median |Δp|": a.median(),
                          "95th pct |Δp|": a.quantile(0.95), "max |Δp|": a.max(),
                          "mean |Δlogit|": d.dlogit.abs().mean(), "share |Δp| < 0.01": float((a < 0.01).mean())}])


def ood_table(rows: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for t, (_, exp) in TESTS.items():
        d = rows[(rows.edit == t) & rows.is_ood]
        if not len(d):
            continue
        for reason, g in d.groupby("ood"):
            r = {"test": t, "OOD reason": reason, "shots": len(g), "mean Δp": g.dp.mean()}
            r["expected direction"] = direction_shares(g.dp, exp)["expected"] if exp in (+1, -1) else np.nan
            recs.append(r)
    return pd.DataFrame(recs)


def _nearest_opponent(s: P.Shot, p) -> float:
    opp = s.ff[(s.ff[:, 2] == 0) & (s.ff[:, 3] == 0)]
    return float(np.hypot(opp[:, 0] - p[0], opp[:, 1] - p[1]).min()) if len(opp) else np.inf


def shooter_move_diagnostics(rows: pd.DataFrame, base: list[P.Shot]) -> pd.DataFrame:
    """1a / 1b: wrong-direction share vs. how the move changes the distance to the nearest
    outfield opponent (the other players stay where they are, so moving the shooter also
    moves them into or out of pressure)."""
    recs = []
    for t in ("1a_closer", "1b_farther"):
        exp = TESTS[t][1]
        d = rows[(rows.edit == t) & ~rows.is_ood]
        change = np.array([_nearest_opponent(base[r.i], EDITS[t](base[r.i]).pos)
                           - _nearest_opponent(base[r.i], base[r.i].pos) for r in d.itertuples()])
        cat = np.select([change < -1, change > 1], ["closer to an opponent (> 1 yd)",
                                                    "away from opponents (> 1 yd)"], "within ±1 yd")
        for c, g in d.groupby(cat):
            sh = direction_shares(g.dp, exp)
            recs.append({"test": t, "nearest outfield opponent": c, "shots": len(g),
                         "expected": sh["expected"], "opposite": sh["opposite"], "mean Δp": g.dp.mean()})
    return pd.DataFrame(recs)


def mid_defender_diagnostics(rows: pd.DataFrame, base: list[P.Shot]) -> pd.DataFrame:
    """2b: share of xG increases vs. the added defender's distance to the goalkeeper."""
    d = rows[(rows.edit == "2b_def_mid") & ~rows.is_ood]
    gk_d = []
    for r in d.itertuples():
        s = base[r.i]
        k = s.gk_index()
        p = P.defender_on_line(s, s.distance / 2)
        gk_d.append(np.hypot(*(s.ff[k, :2] - p)) if k is not None else np.inf)
    g = d.groupby(pd.cut(np.array(gk_d), [0, 3, 6, 10, 20, np.inf], right=False), observed=True)
    out = pd.DataFrame({"shots": g.size(), "share up (wrong)": g.dp.apply(lambda x: (x >= C.SANITY_UNCHANGED).mean()),
                        "share down": g.dp.apply(lambda x: (x <= -C.SANITY_UNCHANGED).mean()),
                        "mean Δp": g.dp.mean()})
    out.index = [f"{iv.left:g}-{iv.right:g} yd" for iv in out.index]
    return out.reset_index(names="added defender to GK")


def failures(rows: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Largest in-distribution changes in the wrong direction, per directional test."""
    recs = []
    for t in DIRECTIONAL:
        exp = TESTS[t][1]
        d = rows[(rows.edit == t) & ~rows.is_ood]
        wrong = d[(np.sign(d.dp) == -exp) & (d.dp.abs() >= C.SANITY_UNCHANGED)]
        recs.append(wrong.assign(score=-exp * wrong.dlogit).nlargest(k, "score"))
    cols = ["edit", "i", "shot_id", "distance", "body_part", "p_orig", "p", "dp", "dlogit", "player_within_1yd"]
    return pd.concat(recs)[cols]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def draw_situation(ax, s: P.Shot, title: str = "", added=None, ghost=None, xlim=(78, 121)):
    """Freeze frame of an (edited) shot; `added` = new defender position, `ghost` = old shooter position."""
    from mplsoccer import Pitch
    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="#888", pitch_color="white")
    pitch.draw(ax=ax)
    ff = s.ff
    team, gk = ff[:, 2] == 1, ff[:, 3] == 1
    pitch.polygon([[(s.x, s.y), (C.GOAL_X, C.POST_LOW_Y), (C.GOAL_X, C.POST_HIGH_Y)]], ax=ax, color="gold", alpha=0.3)
    pitch.scatter(ff[team, 0], ff[team, 1], ax=ax, c="tab:blue", s=45, ec="k", label="teammate")
    pitch.scatter(ff[~team & ~gk, 0], ff[~team & ~gk, 1], ax=ax, c="tab:red", s=45, ec="k", label="opponent")
    pitch.scatter(ff[~team & gk, 0], ff[~team & gk, 1], ax=ax, c="black", s=60, marker="s", label="GK")
    if added is not None:
        pitch.scatter(*added, ax=ax, c="tab:red", s=110, ec="lime", lw=2.5, zorder=6, label="added")
    if ghost is not None:
        pitch.scatter(*ghost, ax=ax, c="none", s=150, marker="*", ec="k", ls="--", zorder=4)
    pitch.scatter(s.x, s.y, ax=ax, c="gold", s=150, marker="*", ec="k", zorder=5, label="shooter")
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=8.5)


def _edit_extras(edit: str, base: P.Shot, t: P.Shot) -> dict:
    if edit.startswith("2"):
        return {"added": t.ff[-1, :2]}
    if edit.startswith("1"):
        return {"ghost": (base.x, base.y)}
    return {}


def plot_effects(rows: pd.DataFrame, bands: pd.DataFrame, path):
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    scored = [_scored_rows(rows, t) for t in TESTS]
    data = [d.dp.to_numpy() for d, _ in scored]
    ax.boxplot(data, whis=(5, 95), showfliers=False, orientation="horizontal",
               tick_labels=[label for _, label in scored])
    ax.axvline(0, color="#999", lw=1)
    for k, t in enumerate(TESTS):
        e = TESTS[t][1]
        if e in (+1, -1):
            ax.annotate("expected ↑" if e > 0 else "expected ↓", (1.0, k + 1), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color="#555")
    ax.invert_yaxis()
    ax.set_xlabel("Δ xG (edited − original); box = quartiles, whiskers = 5th–95th pct")
    ax.set_title("Change in xG per edit (in-distribution shots; 3a is OOD by construction)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    for t in bands.test.unique():
        b = bands[bands.test == t]
        ax2.plot(b.band.astype(str), b.expected, "o-", label=t)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("distance to goal centre of the original shot (yd)")
    ax2.set_ylabel("share in the expected direction")
    ax2.set_title("Expected-direction share by distance", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def pick_examples(df: pd.DataFrame, base: list[P.Shot]) -> list[tuple[str, int]]:
    """Rule-based (not hand-picked) example shots: one random shot per category, fixed seed."""
    rng = np.random.default_rng(C.SANITY_EXAMPLE_SEED)
    cats = [("foot, 12-18 yd", "foot", 12, 18), ("foot, 18-25 yd", "foot", 18, 25),
            ("foot, < 8 yd", "foot", 3, 8), ("header, 6-12 yd", "head", 6, 12)]
    picks = []
    for label, body, lo, hi in cats:
        cand = []
        for i in np.flatnonzero(df.distance.between(lo, hi, inclusive="left").to_numpy()):
            s = base[i]
            if s.body != body or s.ood_reasons():
                continue
            try:
                for e in ("1a_closer", "2a_def_near", "2c_def_aside", "3a_no_gk"):
                    EDITS[e](s)
            except P.NotApplicable:
                continue
            cand.append(i)
        picks.append((label, int(rng.choice(cand))))
    return picks


def plot_examples(df, base, rows, orig, path):
    import matplotlib.pyplot as plt
    picks = pick_examples(df, base)
    cols = ["orig", "1a_closer", "2a_def_near", "2c_def_aside", "3a_no_gk", "body"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3.3 * len(cols), 3.9 * len(picks)))
    for r, (label, i) in enumerate(picks):
        s = base[i]
        p0 = orig.loc[i, "p"]
        for c, e in enumerate(cols):
            if e == "body":
                e = "4a_foot_to_head" if s.body == "foot" else "4b_head_to_foot"
            ax = axes[r, c]
            if e == "orig":
                draw_situation(ax, s, f"{label}\noriginal ({s.body}): xG {p0:.3f}")
                continue
            t = EDITS[e](s)
            row = rows[(rows.i == i) & (rows.edit == e)].iloc[0]
            flag = " [OOD]" if row.is_ood else ""
            draw_situation(ax, t, f"{e}{flag}\nxG {row.p:.3f} (Δ {row.dp:+.3f})", **_edit_extras(e, s, t))
    axes[0, 0].legend(loc="lower left", fontsize=6)
    fig.suptitle("Sanity edits on example test shots (chosen by rule, fixed seed); SmallCNN xG", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def plot_failures(base, fails: pd.DataFrame, path, per_test: int = 1):
    import matplotlib.pyplot as plt
    f = fails.groupby("edit", sort=False).head(per_test)
    n = len(f)
    ncol = 4
    nrow = int(np.ceil(2 * n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow))
    axes = axes.ravel()
    for k, row in enumerate(f.itertuples()):
        s = base[row.i]
        t = EDITS[row.edit](s)
        draw_situation(axes[2 * k], s, f"original ({s.body}, {s.distance:.1f} yd)\nxG {row.p_orig:.3f}")
        draw_situation(axes[2 * k + 1], t, f"{row.edit}: xG {row.p:.3f}\nΔ {row.dp:+.3f} (wrong direction)",
                       **_edit_extras(row.edit, s, t))
    for ax in axes[2 * n:]:
        ax.axis("off")
    fig.suptitle("Largest wrong-direction change per test (in-distribution shots)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def report(split, df, base, rows, orig):
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluate import md_table

    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summ, ctrl, bands = summary_table(rows), control_table(rows), band_table(rows)
    lrt, mir, ood, fails = logreg_table(rows), mirror_table(rows), ood_table(rows), failures(rows)
    gk_fwd = descriptive_table(rows, "3c_gk_forward")
    diag_move, diag_mid = shooter_move_diagnostics(rows, base), mid_defender_diagnostics(rows, base)
    summ.to_csv(C.RESULTS_DIR / f"sanity_{split}.csv", index=False)

    plot_effects(rows, bands, C.PLOTS_DIR / f"sanity_effects_{split}.png")
    plot_examples(df, base, rows, orig, C.PLOTS_DIR / f"sanity_examples_{split}.png")
    plot_failures(base, fails, C.PLOTS_DIR / f"sanity_failures_{split}.png")

    band_wide = bands.pivot(index="test", columns="band", values="expected")
    band_wide = band_wide.reindex(index=bands.test.unique(), columns=band_labels())
    band_wide.columns = [f"{c} yd" for c in band_wide.columns]
    L = [f"# Sanity tests of {C.DISPLAY_NAMES[C.SANITY_MODEL]} on the {split} split", "",
         f"{len(df):,} open-play shots; model `{C.SANITY_MODEL}` (seed 42). Each shot is edited, re-rendered and "
         f"re-scored; Δ = edited − original. |Δp| < {C.SANITY_UNCHANGED} counts as no change. "
         "Shares and means are over in-distribution rows; OOD rows (no goalkeeper, headers from 25+ yd) are "
         "counted in the `OOD` column and reported separately below. Generated by "
         f"`python -m src.sanity --split {split}`.", "",
         "## Summary", "", md_table(summ, floatfmt=".3f"), "",
         "## Control defender (2c)", "",
         "Criterion: per shot, the defender beside the shot line changes xG less than the same defender "
         "in front of the shooter (2a) and halfway along the line (2b).", "",
         md_table(ctrl, floatfmt=".3f"), "",
         "## Expected-direction share by distance band (original shot distance)", "",
         md_table(band_wide.reset_index(), floatfmt=".3f"), "",
         "## CNN vs. LogReg (distance + angle + body part) on the edits it can see", "",
         md_table(lrt, floatfmt=".3f"), "",
         "## Goalkeeper 3 yd forward (3c, descriptive, no expected direction)", "",
         md_table(gk_fwd, floatfmt=".3f"), "",
         "## Mirror symmetry (5)", "",
         "The input of the mirrored shot is exactly the row-flipped tensor, so any difference is an asymmetry "
         "learned by the model.", "",
         md_table(mir, floatfmt=".4f"), "",
         "## Unexpected behaviour", "",
         "### Out-of-distribution situations", "",
         "These edits create situations (almost) absent from training; results here say little about "
         "the model's football logic.", "",
         md_table(ood, floatfmt=".3f") if len(ood) else "(none)", "",
         "### In-distribution: moving the shooter also moves them relative to the defenders", "",
         "The other players stay in place, so a 3 yd move can bring the shooter into (or out of) a "
         "defender's reach. Wrong-direction changes concentrate where the move changes the distance to "
         "the nearest outfield opponent:", "",
         md_table(diag_move, floatfmt=".3f"), "",
         "### In-distribution: a defender halfway along the shot line sometimes raises xG", "",
         md_table(diag_mid, floatfmt=".3f"), "",
         "### In-distribution: largest changes in the wrong direction", "",
         md_table(fails, floatfmt=".3f"), "",
         f"![effects](plots/sanity_effects_{split}.png)", "",
         f"![examples](plots/sanity_examples_{split}.png)", "",
         f"![failures](plots/sanity_failures_{split}.png)", ""]
    out = C.RESULTS_DIR / f"sanity_{split}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out}")
    print(md_table(summ, floatfmt=".3f"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None, help="first N shots only (smoke test)")
    args = ap.parse_args()
    if args.limit:
        _load = load_shots
        load_shots = lambda split=None, **kw: _load(split, **kw).head(args.limit) if split == args.split else _load(split, **kw)  # noqa: E731
    run(args.split)
