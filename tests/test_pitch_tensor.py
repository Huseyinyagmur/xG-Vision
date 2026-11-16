"""Tests for the pitch-tensor renderer.

Run with:  python -m pytest tests -q
Tests that need data/shots.pkl are skipped when it is missing.
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.pitch_tensor import GOAL_DISTANCE_MAP, GRID_X, GRID_Y, mirror_y, render_shot

N_REAL = 2000   # real shots compared per test


# ---------------------------------------------------------------------------
# Reference: the original full-grid implementation (step 1, commit 0eff913)
# ---------------------------------------------------------------------------
def _ref_blobs(points, sigma=C.PLAYER_SIGMA):
    if len(points) == 0:
        return np.zeros((C.IMG_H, C.IMG_W), np.float32)
    dx = GRID_X[None] - points[:, 0, None, None]
    dy = GRID_Y[None] - points[:, 1, None, None]
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2)).sum(0).astype(np.float32)


def _ref_cone(sx, sy):
    def side(px, py, qx, qy, rx, ry):
        return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)

    d1 = side(GRID_X, GRID_Y, sx, sy, C.GOAL_X, C.POST_LOW_Y)
    d2 = side(GRID_X, GRID_Y, C.GOAL_X, C.POST_LOW_Y, C.GOAL_X, C.POST_HIGH_Y)
    d3 = side(GRID_X, GRID_Y, C.GOAL_X, C.POST_HIGH_Y, sx, sy)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return (~(neg & pos)).astype(np.float32)


def _ref_render(sx, sy, ff):
    ff = np.asarray(ff, dtype=np.float32).reshape(-1, 4)
    ff = ff[(ff[:, 0] >= C.X_MIN) & (ff[:, 0] <= C.X_MAX)]
    team, gk = ff[:, 2] == 1, ff[:, 3] == 1
    out = np.empty((C.N_CHANNELS, C.IMG_H, C.IMG_W), np.float32)
    out[0] = _ref_blobs(np.array([[sx, sy]], np.float32))
    out[1] = _ref_blobs(ff[team, :2])
    out[2] = _ref_blobs(ff[~team & ~gk, :2])
    out[3] = _ref_blobs(ff[~team & gk, :2])
    out[4] = _ref_cone(sx, sy)
    out[5] = GOAL_DISTANCE_MAP
    return out


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_shots():
    if not C.SHOTS_PATH.exists():
        pytest.skip("data/shots.pkl not found")
    df = pd.read_pickle(C.SHOTS_PATH)
    return df.sample(N_REAL, random_state=0)


def _edge_cases():
    """Synthetic shots at the borders of the rendered region."""
    ff = np.array([
        [60.0, 0.0, 1, 0], [120.0, 80.0, 0, 0], [120.0, 40.0, 0, 1],
        [60.2, 79.9, 0, 0], [119.9, 0.1, 1, 0], [59.0, 40.0, 0, 0],  # last one is outside
    ], np.float32)
    shooters = [(60.0, 0.0), (120.0, 40.0), (120.0, 0.0), (119.5, 80.0), (102.0, 40.0), (60.0, 80.0)]
    return [(sx, sy, ff) for sx, sy in shooters]


def _assert_cone_matches(new, ref, sx, sy):
    """Cones may only differ on pixels whose centre lies exactly on a triangle edge
    (the new renderer counts those as inside, the reference left it to rounding)."""
    for i, j in np.argwhere(new != ref):
        px, py = GRID_X[i, j], GRID_Y[i, j]
        on_edge = False
        for (ax, ay), (bx, by) in [((sx, sy), (C.GOAL_X, C.POST_LOW_Y)),
                                   ((sx, sy), (C.GOAL_X, C.POST_HIGH_Y)),
                                   ((C.GOAL_X, C.POST_LOW_Y), (C.GOAL_X, C.POST_HIGH_Y))]:
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            on_edge |= abs(cross) / max(np.hypot(bx - ax, by - ay), 1e-9) < 1e-6
        assert on_edge, (sx, sy, px, py)


def test_matches_reference_on_edge_cases():
    for sx, sy, ff in _edge_cases():
        new, ref = render_shot(sx, sy, ff), _ref_render(sx, sy, ff)
        assert np.abs(new[:4] - ref[:4]).max() < 1e-3, (sx, sy)
        _assert_cone_matches(new[4], ref[4], sx, sy)


def test_matches_reference_on_real_shots(real_shots):
    worst = 0.0
    for r in real_shots.itertuples():
        new, ref = render_shot(r.x, r.y, r.freeze_frame), _ref_render(r.x, r.y, r.freeze_frame)
        _assert_cone_matches(new[4], ref[4], r.x, r.y)
        np.testing.assert_array_equal(new[5], ref[5])
        worst = max(worst, float(np.abs(new[:4] - ref[:4]).max()))
    assert worst < 1e-3, worst                                  # blobs: truncation error only


def test_mirror_equals_row_flip(real_shots):
    for r in real_shots.itertuples():
        orig = render_shot(r.x, r.y, r.freeze_frame)
        msy, mff = mirror_y(r.y, r.freeze_frame)
        mirrored = render_shot(r.x, msy, mff)
        flipped = orig[:, ::-1, :]
        np.testing.assert_allclose(mirrored[:4], flipped[:4], atol=1e-5)
        np.testing.assert_array_equal(mirrored[4], flipped[4])
        np.testing.assert_array_equal(mirrored[5], flipped[5])
