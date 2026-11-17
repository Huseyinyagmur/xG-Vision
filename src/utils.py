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


def free_memory_gb() -> tuple[float, float] | None:
    """(available physical RAM, available commit charge) in GB, or None if unknown.
    On Windows the commit limit (RAM + page file) is what DataLoader workers run out of first
    (error 1455), so both numbers matter."""
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return m.ullAvailPhys / 2**30, m.ullAvailPageFile / 2**30
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
        return avail, avail
    except (ValueError, OSError, AttributeError):
        return None


def pick_num_workers(default: int, low: int = 2, min_free_gb: float = 6.0) -> int:
    """`default` DataLoader workers, or `low` if less than `min_free_gb` of RAM / commit is free
    (e.g. a game is running next to the job)."""
    mem = free_memory_gb()
    if mem is None:
        return default
    phys, commit = mem
    n = default if min(phys, commit) >= min_free_gb else min(low, default)
    print(f"free memory: {phys:.1f} GB RAM, {commit:.1f} GB commit -> num_workers = {n}")
    return n


def is_memory_error(err: BaseException) -> bool:
    """Out-of-memory failures of DataLoader workers on Windows look like this."""
    if isinstance(err, MemoryError):
        return True
    msg = str(err).lower()
    return any(k in msg for k in ("1455", "paging file", "not enough memory", "out of memory",
                                  "worker", "cannot allocate memory"))


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
