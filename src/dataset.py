"""PyTorch Dataset that renders pitch tensors on the fly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config as C
from src.data_download import training_subset
from src.pitch_tensor import mirror_y, render_shot


def load_shots(split: str | None = None, path=C.SHOTS_PATH) -> pd.DataFrame:
    df = training_subset(pd.read_pickle(path))
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


class ShotDataset(Dataset):
    """
    Returns (tensor[C,H,W], label float, index). `augment=True` mirrors the
    pitch left/right with probability 0.5 (use only for the training split).
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.sx = df["x"].to_numpy(np.float32)
        self.sy = df["y"].to_numpy(np.float32)
        self.ff = df["freeze_frame"].tolist()
        self.y = df["is_goal"].to_numpy(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        sx, sy, ff = self.sx[i], self.sy[i], self.ff[i]
        if self.augment and np.random.rand() < 0.5:
            sy, ff = mirror_y(sy, ff)
        x = torch.from_numpy(render_shot(sx, sy, ff))
        return x, torch.tensor(self.y[i]), i
