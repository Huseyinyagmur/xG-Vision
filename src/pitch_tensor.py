"""Render a shot (shooter + freeze frame) into a multi-channel bird's-eye tensor.

Tensor layout: (C, H, W) with
    rows (H)    <-> pitch y in [Y_MIN, Y_MAX]
    columns (W) <-> pitch x in [X_MIN, X_MAX]   (goal on the right edge)
Row 0 is y = 0, which is the TOP of a StatsBomb / mplsoccer pitch, so
`imshow(t[c], origin="upper", extent=[X_MIN, X_MAX, Y_MAX, Y_MIN])` overlays 1:1.

Rendering is cheap (~1.2 ms per shot in a single process on a laptop CPU; the
original full-grid version took ~10-14 ms), so tensors are produced on the fly
inside the Dataset instead of being stored (all shots x 6 x 160 x 120 float32
would be tens of GB). Speed comes from two tricks:
  * each player's Gaussian is only evaluated in a small window around it
    (truncated at BLOB_TRUNCATE sigmas) and is separable (outer product of a
    1-D row and a 1-D column profile);
  * the shot cone is only evaluated inside its bounding box.
"""
from __future__ import annotations

import math

import numpy as np

from src import config as C

# Pixel-centre coordinates in pitch units
_xs = (C.X_MIN + (np.arange(C.IMG_W) + 0.5) / C.PX_PER_YARD).astype(np.float64)
_ys = (C.Y_MIN + (np.arange(C.IMG_H) + 0.5) / C.PX_PER_YARD).astype(np.float64)
GRID_X, GRID_Y = np.meshgrid(_xs, _ys)                     # (H, W)

# Static goal-distance map, normalised to [0, 1] (1 = on the goal line)
_d = np.hypot(C.GOAL_CENTER[0] - GRID_X, C.GOAL_CENTER[1] - GRID_Y)
GOAL_DISTANCE_MAP = (1.0 - _d / _d.max()).astype(np.float32)

# Gaussians are cut off at this many sigmas (exp(-4**2 / 2) ~ 3e-4).
BLOB_TRUNCATE = 4.0
# Tolerance (yards^2) of the point-in-triangle test in shot_cone.
EDGE_EPS = 1e-6


def gaussian_blobs(points: np.ndarray, sigma: float = C.PLAYER_SIGMA) -> np.ndarray:
    """Sum of isotropic Gaussians (peak 1) centred at `points` (N, 2) -> (H, W)."""
    out = np.zeros((C.IMG_H, C.IMG_W), np.float32)
    reach = BLOB_TRUNCATE * sigma * C.PX_PER_YARD          # cut-off radius (px)
    inv = 1.0 / (2.0 * sigma * sigma)

    def window(p, lo, n):
        # pixels whose centre lies within `reach` of p; symmetric around p, so a
        # mirrored player gets exactly the mirrored window
        u = (p - lo) * C.PX_PER_YARD - 0.5                 # p in pixel-index units
        return max(math.ceil(u - reach), 0), min(math.floor(u + reach) + 1, n)

    for px, py in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        c0, c1 = window(px, C.X_MIN, C.IMG_W)
        r0, r1 = window(py, C.Y_MIN, C.IMG_H)
        if c0 >= c1 or r0 >= r1:
            continue
        gx = np.exp(-((_xs[c0:c1] - px) ** 2) * inv)
        gy = np.exp(-((_ys[r0:r1] - py) ** 2) * inv)
        out[r0:r1, c0:c1] += np.outer(gy, gx).astype(np.float32)
    return out


def shot_cone(sx: float, sy: float) -> np.ndarray:
    """Binary mask of the triangle (shooter, low post, high post)."""
    out = np.zeros((C.IMG_H, C.IMG_W), np.float32)
    ax, ay = sx, sy
    bx, by = C.GOAL_X, C.POST_LOW_Y
    cx, cy = C.GOAL_X, C.POST_HIGH_Y

    # Bounding box of the triangle in pixel indices (one pixel of margin)
    c0 = max(int((min(ax, bx) - C.X_MIN) * C.PX_PER_YARD) - 1, 0)
    c1 = min(int((max(ax, bx) - C.X_MIN) * C.PX_PER_YARD) + 2, C.IMG_W)
    r0 = max(int((min(ay, by) - C.Y_MIN) * C.PX_PER_YARD) - 1, 0)
    r1 = min(int((max(ay, cy) - C.Y_MIN) * C.PX_PER_YARD) + 2, C.IMG_H)
    gx, gy = GRID_X[r0:r1, c0:c1], GRID_Y[r0:r1, c0:c1]

    def side(px, py, qx, qy, rx, ry):
        return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)

    # Pixel centres lying exactly on an edge are counted as inside (tolerance EDGE_EPS);
    # without it, rounding decides and the mirrored shot would get a different cone.
    d1 = side(gx, gy, ax, ay, bx, by)
    d2 = side(gx, gy, bx, by, cx, cy)
    d3 = side(gx, gy, cx, cy, ax, ay)
    neg = (d1 < -EDGE_EPS) | (d2 < -EDGE_EPS) | (d3 < -EDGE_EPS)
    pos = (d1 > EDGE_EPS) | (d2 > EDGE_EPS) | (d3 > EDGE_EPS)
    out[r0:r1, c0:c1] = ~(neg & pos)
    return out


def render_shot(sx: float, sy: float, freeze_frame: np.ndarray) -> np.ndarray:
    """
    sx, sy        : shooter location (StatsBomb yards)
    freeze_frame  : (N, 4) array [x, y, teammate, is_gk]
    returns       : (N_CHANNELS, IMG_H, IMG_W) float32
    """
    ff = np.asarray(freeze_frame, dtype=np.float32).reshape(-1, 4)
    # players outside the rendered region are dropped (they are far from goal anyway)
    ff = ff[(ff[:, 0] >= C.X_MIN) & (ff[:, 0] <= C.X_MAX)]
    team = ff[:, 2] == 1
    gk = ff[:, 3] == 1

    out = np.empty((C.N_CHANNELS, C.IMG_H, C.IMG_W), np.float32)
    out[0] = gaussian_blobs(np.array([[sx, sy]], np.float32))
    out[1] = gaussian_blobs(ff[team, :2])
    out[2] = gaussian_blobs(ff[~team & ~gk, :2])
    out[3] = gaussian_blobs(ff[~team & gk, :2])
    out[4] = shot_cone(sx, sy)
    out[5] = GOAL_DISTANCE_MAP
    return out


def mirror_y(sy: float, freeze_frame: np.ndarray):
    """Left/right mirror (y -> 80 - y). The goal is symmetric, so this is a valid augmentation."""
    ff = np.array(freeze_frame, dtype=np.float32, copy=True).reshape(-1, 4)
    ff[:, 1] = C.PITCH_WIDTH - ff[:, 1]
    return C.PITCH_WIDTH - sy, ff
