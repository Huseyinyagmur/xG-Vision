"""Worker count selection and the out-of-memory retry (no data needed)."""
import numpy as np
import torch

from src import sanity, utils


def test_pick_num_workers(monkeypatch):
    monkeypatch.setattr(utils, "free_memory_gb", lambda: (12.0, 10.0))
    assert utils.pick_num_workers(6) == 6
    monkeypatch.setattr(utils, "free_memory_gb", lambda: (12.0, 3.0))     # commit is the limit
    assert utils.pick_num_workers(6) == 2
    monkeypatch.setattr(utils, "free_memory_gb", lambda: None)
    assert utils.pick_num_workers(6) == 6


def test_free_memory_reports_numbers():
    mem = utils.free_memory_gb()
    assert mem is None or (mem[0] > 0 and mem[1] > 0)


def test_predict_retries_with_fewer_workers(monkeypatch):
    calls = []

    def fake(model, ds, device, batch_size, num_workers):
        calls.append((batch_size, num_workers))
        if num_workers > 1:
            raise RuntimeError("DataLoader worker (pid 123) exited unexpectedly: The paging file is too small (1455)")
        return np.zeros(3)

    monkeypatch.setattr(sanity, "_predict_once", fake)
    out = sanity.predict_logits(None, None, torch.device("cpu"), batch_size=512, num_workers=6)
    assert calls == [(512, 6), (256, 3), (128, 1)] and out.shape == (3,)


def test_predict_does_not_swallow_other_errors(monkeypatch):
    def fake(*a, **k):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(sanity, "_predict_once", fake)
    try:
        sanity.predict_logits(None, None, torch.device("cpu"), num_workers=2)
    except RuntimeError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("expected the error to propagate")
