"""Render a shot (shooter + freeze frame) into a multi-channel bird's-eye tensor.

Tensor layout: (C, H, W) with
    rows (H)    <-> pitch y in [Y_MIN, Y_MAX]
    columns (W) <-> pitch x in [X_MIN, X_MAX]   (goal on the right edge)
Row 0 is y = 0, which is the TOP of a StatsBomb / mplsoccer pitch, so
`imshow(t[c], origin="upper", extent=[X_MIN, X_MAX, Y_MAX, Y_MIN])` overlays 1:1.

Rendering is cheap (~1 ms), so tensors are produced on the fly inside the
Dataset instead of being stored (all shots x 6 x 160 x 120 float32 would be
tens of GB).
"""
from __future__ import annotations

import numpy as np

from src import config as C

# Pixel-centre coordinates in pitch units
_xs = C.X_MIN + (np.arange(C.IMG_W) + 0.5) / C.PX_PER_YARD
_ys = C.Y_MIN + (np.arange(C.IMG_H) + 0.5) / C.PX_PER_YARD
GRID_X, GRID_Y = np.meshgrid(_xs, _ys)                     # (H, W)

# Static goal-distance map, normalised to [0, 1] (1 = on the goal line)
_d = np.hypot(C.GOAL_CENTER[0] - GRID_X, C.GOAL_CENTER[1] - GRID_Y)
GOAL_DISTANCE_MAP = (1.0 - _d / _d.max()).astype(np.float32)


def gaussian_blobs(points: np.ndarray, sigma: float = C.PLAYER_SIGMA) -> np.ndarray:
    """Sum of isotropic Gaussians (peak 1) centred at `points` (N, 2) -> (H, W)."""
    if len(points) == 0:
        return np.zeros((C.IMG_H, C.IMG_W), np.float32)
    dx = GRID_X[None] - points[:, 0, None, None]
    dy = GRID_Y[None] - points[:, 1, None, None]
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2)).sum(0).astype(np.float32)


def shot_cone(sx: float, sy: float) -> np.ndarray:
    """Binary mask of the triangle (shooter, low post, high post)."""
    ax, ay = sx, sy
    bx, by = C.GOAL_X, C.POST_LOW_Y
    cx, cy = C.GOAL_X, C.POST_HIGH_Y

    def side(px, py, qx, qy, rx, ry):
        return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)

    d1 = side(GRID_X, GRID_Y, ax, ay, bx, by)
    d2 = side(GRID_X, GRID_Y, bx, by, cx, cy)
    d3 = side(GRID_X, GRID_Y, cx, cy, ax, ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    mask = ~(neg & pos)
    # make sure very thin cones (shooter near the byline) still show the shooter pixel
    return mask.astype(np.float32)


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
