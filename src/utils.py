"""Small helpers: seeding, device selection, logging to a file."""
from __future__ import annotations

import os
import random
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

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


class _Tee:
    """File-like object that writes to several streams (used to mirror stdout into a log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_output(log_path):
    """Mirror stdout and stderr into `log_path` (overwritten) while the block runs."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, f), _Tee(old_err, f)
        try:
            yield
        except BaseException:
            traceback.print_exc()          # keep the traceback in the log as well
            raise
        finally:
            sys.stdout, sys.stderr = old_out, old_err
