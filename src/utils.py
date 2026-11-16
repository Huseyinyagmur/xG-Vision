"""Small helpers: seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed python, numpy and torch. Returns a torch.Generator for DataLoader shuffling.

    DataLoader workers get their numpy seed from torch's per-worker seed (torch >= 1.9),
    so augmentation is reproducible for a fixed seed and num_workers. cuDNN autotuning
    stays on for speed, so GPU results can differ in the last digits between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return torch.Generator().manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
