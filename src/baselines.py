"""Logistic-regression baselines on hand-crafted features, fitted on the training split.

    logreg_dist_angle       : distance to goal centre + opening angle of the goal mouth
    logreg_dist_angle_body  : the same + body part one-hot (head / foot / other, as the CNNs get it)

They are cheap and deterministic, so they are refitted whenever predictions are needed
instead of being saved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.features import body_part_onehot

LOGREG_FEATURES = {
    "logreg_dist_angle": False,          # name -> include body part?
    "logreg_dist_angle_body": True,
}


def design_matrix(df: pd.DataFrame, body_part: bool) -> np.ndarray:
    x = df[["distance", "angle"]].to_numpy(np.float64)
    if body_part:
        x = np.hstack([x, body_part_onehot(df["body_part"]).astype(np.float64)])
    return x


def fit_logreg(train_df: pd.DataFrame, body_part: bool):
    """Standardised features + logistic regression (default, weak L2; ~67k shots)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=C.SEED))
    model.fit(design_matrix(train_df, body_part), train_df["is_goal"].to_numpy())
    return model


def logreg_predictions(train_df: pd.DataFrame, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """P(goal) of every logistic-regression baseline for the shots in `df`."""
    out = {}
    for name, body in LOGREG_FEATURES.items():
        model = fit_logreg(train_df, body)
        out[name] = model.predict_proba(design_matrix(df, body))[:, 1]
    return out


if __name__ == "__main__":
    # python -m src.baselines  -> coefficients and validation metrics
    from src.dataset import load_shots
    from src.evaluate import compute_metrics, format_table

    tr, va = load_shots("train"), load_shots("val")
    for name, body in LOGREG_FEATURES.items():
        m = fit_logreg(tr, body)
        cols = ["distance", "angle"] + ([f"body_{g}" for g in C.BODY_PART_GROUPS] if body else [])
        coef = {c: round(float(v), 3) for c, v in zip(cols, m[-1].coef_[0])}
        print(f"{name}: intercept {m[-1].intercept_[0]:.3f}, standardised coefs {coef}")
    preds = logreg_predictions(tr, va)
    print("\n[val]")
    print(format_table({k: compute_metrics(va["is_goal"], p) for k, p in preds.items()}))
