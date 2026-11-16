"""PhaseBatchSampler tests (no data needed)."""
import numpy as np

from src.dataset import PhaseBatchSampler


def _sampler(seed=0):
    return PhaseBatchSampler({
        "train": (np.arange(10), True, None),
        "bn": (np.arange(10), True, 2),
        "val": (np.arange(10, 15), False, None),
    }, batch_size=4, seed=seed)


def test_phase_lengths_and_contents():
    s = _sampler()
    s.phase = "train"
    tr = list(s)
    assert len(s) == len(tr) == 2                       # drop_last: 10 // 4
    assert all(len(b) == 4 for b in tr)
    assert {i for b in tr for i, _ in b} <= set(range(10))
    s.phase = "bn"
    assert len(list(s)) == 2
    s.phase = "val"
    va = list(s)
    assert len(s) == len(va) == 2                       # keeps the last, smaller batch
    assert [i for b in va for i, _ in b] == list(range(10, 15))   # ordered, complete
    assert not any(m for b in va for _, m in b)         # never mirrored


def test_train_phase_is_reproducible_and_reshuffled():
    a, b = _sampler(seed=1), _sampler(seed=1)
    a.phase = b.phase = "train"
    ep1_a, ep1_b = list(a), list(b)
    assert ep1_a == ep1_b                                # same seed -> same order and mirroring
    assert list(a) != ep1_a                              # next epoch is reshuffled
