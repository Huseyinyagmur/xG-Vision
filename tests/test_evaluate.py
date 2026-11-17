"""Evaluation helpers against sklearn (no data needed)."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.evaluate import _weighted_auc, bootstrap_by_match, reliability_bins


def _toy(n=600, n_matches=40, seed=0):
    rng = np.random.default_rng(seed)
    # rounding -> many tied scores; kept away from 0 so log-loss clipping conventions do not matter
    p = np.clip(np.round(rng.beta(1, 8, n), 2), 0.01, 0.99)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"match_id": rng.integers(0, n_matches, n), "is_goal": y, "a": p,
                         "b": np.clip(p + rng.normal(0, 0.05, n), 0.001, 0.999)})


def test_weighted_auc_matches_sklearn_with_ties():
    df = _toy()
    p, y = df["a"].to_numpy(), df["is_goal"].to_numpy().astype(float)
    w = np.random.default_rng(1).integers(0, 4, (3, len(p))).astype(float)
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    ours = _weighted_auc(ps, y[order], starts, w[:, order])
    ref = [roc_auc_score(y, p, sample_weight=wi) for wi in w]
    np.testing.assert_allclose(ours, ref, rtol=1e-10)


def test_bootstrap_equals_metrics_on_resampled_matches():
    df = _toy()
    res = bootstrap_by_match(df, ["a"], B=3, seed=5)["a"]
    # rebuild resample 0 by hand: draw the same matches and stack their shots
    codes, _ = pd.factorize(df["match_id"])
    rng = np.random.default_rng(5)
    draw = rng.integers(0, codes.max() + 1, codes.max() + 1)
    rows = np.concatenate([np.flatnonzero(codes == m) for m in draw])
    y, p = df["is_goal"].to_numpy()[rows], df["a"].to_numpy()[rows]
    np.testing.assert_allclose(res[0], [log_loss(y, p), brier_score_loss(y, p), roc_auc_score(y, p)],
                               rtol=1e-9)


def test_reliability_bins_equal_count_and_rate():
    df = _toy(n=1500)
    bins = reliability_bins(df["a"], df["is_goal"], df["match_id"], n_bins=10, B=20)
    assert bins["n"].sum() == 1500 and bins["n"].min() == bins["n"].max() == 150
    assert np.isclose((bins["goal_rate"] * bins["n"]).sum() / 1500, df["is_goal"].mean())
    assert (bins["mean_pred"].diff().dropna() >= 0).all()
