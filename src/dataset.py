"""PyTorch Dataset that renders pitch tensors on the fly, plus a phase-aware batch sampler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src import config as C
from src.data_download import training_subset
from src.features import extra_features
from src.pitch_tensor import mirror_y, render_shot


def load_shots(split: str | None = None, path=C.SHOTS_PATH) -> pd.DataFrame:
    df = training_subset(pd.read_pickle(path))
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


class ShotDataset(Dataset):
    """
    Returns (image[C,H,W], extra[N_EXTRA_FEATURES], label float, index).

    The key may be an int `i` or a tuple `(i, mirror)`. With an int key,
    `augment=True` mirrors the pitch left/right with probability 0.5 (use only for
    the training split). With a tuple key the caller decides (see PhaseBatchSampler).
    The extra features (body part head/foot/other) are mirror-invariant.

    `image_dtype=torch.float16` halves the memory that DataLoader workers pass
    to the main process (a float32 batch of 256 is ~118 MB); cast back on the GPU.
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False, image_dtype: torch.dtype = torch.float32):
        self.sx = df["x"].to_numpy(np.float32)
        self.sy = df["y"].to_numpy(np.float32)
        self.ff = df["freeze_frame"].tolist()
        self.extra = torch.from_numpy(extra_features(df))
        self.y = df["is_goal"].to_numpy(np.float32)
        self.augment = augment
        self.image_dtype = image_dtype

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, mirror = key
        else:
            i, mirror = key, self.augment and np.random.rand() < 0.5
        sx, sy, ff = self.sx[i], self.sy[i], self.ff[i]
        if mirror:
            sy, ff = mirror_y(sy, ff)
        x = torch.from_numpy(render_shot(sx, sy, ff)).to(self.image_dtype)
        return x, self.extra[i], torch.tensor(self.y[i]), i


class PhaseBatchSampler(Sampler):
    """
    Batch sampler over a dataset that holds several splits, switched by `.phase`.

    Lets one DataLoader with persistent workers serve training and validation:
    on Windows every worker is a fresh process (~10 s start-up each), so starting
    them once per run instead of twice per epoch matters. The sampler runs in the
    main process and yields `(index, mirror)` keys, so the augmentation is drawn
    from one seeded generator and does not depend on the number of workers.

    phases: name -> (dataset indices, shuffle+augment?, max batches or None)
    """

    def __init__(self, phases: dict[str, tuple[np.ndarray, bool, int | None]], batch_size: int, seed: int):
        self.phases = {k: (np.asarray(idx), train, cap) for k, (idx, train, cap) in phases.items()}
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.phase = next(iter(self.phases))

    def __len__(self):
        idx, train, cap = self.phases[self.phase]
        n = len(idx) // self.batch_size if train else -(-len(idx) // self.batch_size)
        return n if cap is None else min(n, cap)

    def __iter__(self):
        idx, train, _ = self.phases[self.phase]
        if train:
            idx = self.rng.permutation(idx)
            mirror = self.rng.random(len(idx)) < 0.5
        else:
            mirror = np.zeros(len(idx), bool)
        for b in range(len(self)):
            sl = slice(b * self.batch_size, (b + 1) * self.batch_size)
            yield [(int(i), bool(m)) for i, m in zip(idx[sl], mirror[sl])]
