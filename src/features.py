"""Scalar ("extra") features fed to the model next to the pitch image."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def body_part_onehot(body_part: pd.Series) -> np.ndarray:
    """StatsBomb body part -> one-hot over BODY_PART_GROUPS (left/right foot merged)."""
    groups = body_part.map(C.BODY_PART_MAP)
    if groups.isna().any():
        unknown = sorted(body_part[groups.isna()].unique())
        raise ValueError(f"unknown body part(s): {unknown}")
    idx = groups.map({g: i for i, g in enumerate(C.BODY_PART_GROUPS)}).to_numpy()
    return np.eye(len(C.BODY_PART_GROUPS), dtype=np.float32)[idx]


def extra_features(df: pd.DataFrame) -> np.ndarray:
    """(N, N_EXTRA_FEATURES) float32 matrix of the scalar model inputs."""
    feats = body_part_onehot(df["body_part"])
    assert feats.shape[1] == C.N_EXTRA_FEATURES
    return feats
