"""Probability metrics shared by every model (CNNs, baselines, StatsBomb xG).

    python -m src.evaluate      # reference metrics (StatsBomb xG, constant) on val and test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-7


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


def main():
    from src.dataset import load_shots

    train_rate = load_shots("train")["is_goal"].mean()
    for split in ("val", "test"):
        df = load_shots(split)
        res = {name: compute_metrics(df["is_goal"], p)
               for name, p in reference_predictions(df, train_rate).items()}
        print(f"\n[{split}] n = {len(df)}, goal rate {df['is_goal'].mean():.4f}")
        print(format_table(res))


if __name__ == "__main__":
    main()
