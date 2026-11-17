"""Controlled edits of a shot situation, for the sanity tests and heat maps (step 4).

A shot is (shooter x, shooter y, freeze frame, body part group). Every function returns a new
`Shot` and never modifies its input. Geometry is in StatsBomb yards; the attacked goal is at
x = 120 with its centre at y = 40. The "shot line" runs from the shooter to the goal centre.

Functions that cannot be applied to a shot (e.g. moving a shooter 3 yd closer who is 2 yd
from goal) raise `NotApplicable`; the caller skips such shots and reports how many remain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src import config as C
from src.pitch_tensor import mirror_y, render_shot

GOAL = np.array(C.GOAL_CENTER, dtype=np.float64)
OOD_HEADER_DISTANCE = 25.0          # headers from 25+ yd are (almost) absent from the training data


class NotApplicable(ValueError):
    """The edit makes no sense for this shot (it would leave the pitch, the player is missing...)."""


@dataclass(frozen=True)
class Shot:
    x: float
    y: float
    ff: np.ndarray                  # (N, 4) [x, y, teammate, is_gk], shooter not included
    body: str                       # one of C.BODY_PART_GROUPS

    @classmethod
    def from_row(cls, row) -> "Shot":
        return cls(float(row.x), float(row.y), np.asarray(row.freeze_frame, np.float32).reshape(-1, 4),
                   C.BODY_PART_MAP[row.body_part])

    # --- model inputs -------------------------------------------------------
    def image(self) -> np.ndarray:
        return render_shot(self.x, self.y, self.ff)

    def extra(self) -> np.ndarray:
        e = np.zeros(C.N_EXTRA_FEATURES, np.float32)
        e[C.BODY_PART_GROUPS.index(self.body)] = 1.0
        return e

    # --- geometry -----------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], np.float64)

    @property
    def distance(self) -> float:
        return float(np.hypot(*(GOAL - self.pos)))

    @property
    def angle(self) -> float:
        return float(distance_angle(self.x, self.y)[1])

    def line_dir(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vector shooter -> goal centre, and its left normal."""
        d = GOAL - self.pos
        n = np.linalg.norm(d)
        if n < 1e-6:
            raise NotApplicable("shooter is on the goal centre")
        u = d / n
        return u, np.array([-u[1], u[0]])

    def gk_index(self) -> int | None:
        """Row of the defending goalkeeper in the freeze frame (None if absent)."""
        idx = np.flatnonzero((self.ff[:, 2] == 0) & (self.ff[:, 3] == 1))
        return int(idx[0]) if len(idx) else None

    # --- out-of-distribution flags --------------------------------------------
    def ood_reasons(self) -> list[str]:
        """Situations (almost) never seen in training: no goalkeeper, long-range headers."""
        reasons = []
        if self.gk_index() is None:
            reasons.append("no goalkeeper")
        if self.body == "head" and self.distance >= OOD_HEADER_DISTANCE:
            reasons.append(f"header from {OOD_HEADER_DISTANCE:.0f}+ yd")
        return reasons


def distance_angle(x, y):
    """Distance to the goal centre and opening angle of the goal mouth (rad), as in data_download."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    dx, dy = C.GOAL_X - x, C.GOAL_CENTER[1] - y
    a1 = np.arctan2(C.POST_LOW_Y - y, dx)
    a2 = np.arctan2(C.POST_HIGH_Y - y, dx)
    return np.hypot(dx, dy), np.abs(a2 - a1)


def in_cone(px: float, py: float, sx: float, sy: float) -> bool:
    """Is (px, py) inside the triangle shooter -> both posts (edges count as inside)?"""
    a, b, c = (sx, sy), (C.GOAL_X, C.POST_LOW_Y), (C.GOAL_X, C.POST_HIGH_Y)

    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2, d3 = cross(a, b, (px, py)), cross(b, c, (px, py)), cross(c, a, (px, py))
    has_neg = min(d1, d2, d3) < -1e-9
    has_pos = max(d1, d2, d3) > 1e-9
    return not (has_neg and has_pos)


def _inside_region(p) -> bool:
    return C.X_MIN <= p[0] <= C.X_MAX and C.Y_MIN <= p[1] <= C.Y_MAX


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------
def move_shooter(shot: Shot, d: float, min_distance: float = 2.0) -> Shot:
    """Move the shooter d yards along the shot line (d > 0: towards goal, d < 0: away).
    Everyone else stays where they are."""
    u, _ = shot.line_dir()
    if d > 0 and shot.distance - d < min_distance:
        raise NotApplicable("would end up on top of the goal")
    p = shot.pos + d * u
    if not _inside_region(p):
        raise NotApplicable("would leave the rendered half")
    return replace(shot, x=float(p[0]), y=float(p[1]))


def _with_player(shot: Shot, p, teammate: int = 0, is_gk: int = 0) -> Shot:
    row = np.array([[p[0], p[1], teammate, is_gk]], np.float32)
    return replace(shot, ff=np.vstack([shot.ff, row]))


def defender_on_line(shot: Shot, along: float) -> np.ndarray:
    """Point `along` yards from the shooter on the shot line."""
    u, _ = shot.line_dir()
    if not 0 < along < shot.distance:
        raise NotApplicable("point is not between shooter and goal")
    return shot.pos + along * u


def add_defender_on_line(shot: Shot, along: float) -> Shot:
    """Outfield defender `along` yards in front of the shooter, on the shot line (inside the cone)."""
    return _with_player(shot, defender_on_line(shot, along))


def control_point(shot: Shot, along: float, offset: float) -> np.ndarray:
    """Point `along` yards down the shot line, moved `offset` yards sideways (perpendicular).
    The side is the one towards the pitch's middle (y = 40), left normal on a tie.
    Raises NotApplicable if the point still lies in the shot cone or off the rendered half."""
    _, n = shot.line_dir()
    base = defender_on_line(shot, along)
    cands = [base + offset * n, base - offset * n]
    cands.sort(key=lambda p: abs(p[1] - GOAL[1]))     # stable: left normal first on a tie
    for p in cands:
        if _inside_region(p) and not in_cone(p[0], p[1], shot.x, shot.y):
            return p
    raise NotApplicable("no side point outside the cone")


def add_defender_aside(shot: Shot, along: float, offset: float) -> Shot:
    """Control edit: the same defender, but beside the shot line and outside the cone."""
    return _with_player(shot, control_point(shot, along, offset))


def remove_goalkeeper(shot: Shot) -> Shot:
    k = shot.gk_index()
    if k is None:
        raise NotApplicable("no goalkeeper")
    return replace(shot, ff=np.delete(shot.ff, k, axis=0))


def _move_gk(shot: Shot, new_pos) -> Shot:
    """Not clipped: a goalkeeper pushed behind the goal line would be a different edit
    (sideways + backwards), so such shots are skipped instead."""
    if not _inside_region(new_pos):
        raise NotApplicable("goalkeeper would leave the pitch")
    k = shot.gk_index()
    ff = shot.ff.copy()
    ff[k, :2] = new_pos
    return replace(shot, ff=ff)


def shift_goalkeeper_sideways(shot: Shot, d: float) -> Shot:
    """Move the goalkeeper d yards perpendicular to the shot line, away from that line.
    A goalkeeper exactly on the line is moved to the left normal side."""
    k = shot.gk_index()
    if k is None:
        raise NotApplicable("no goalkeeper")
    _, n = shot.line_dir()
    g = shot.ff[k, :2].astype(np.float64)
    side = np.sign(np.dot(g - shot.pos, n)) or 1.0
    return _move_gk(shot, g + side * d * n)


def shift_goalkeeper_forward(shot: Shot, d: float) -> Shot:
    """Move the goalkeeper d yards straight towards the shooter (narrowing the angle)."""
    k = shot.gk_index()
    if k is None:
        raise NotApplicable("no goalkeeper")
    g = shot.ff[k, :2].astype(np.float64)
    v = shot.pos - g
    dist = np.linalg.norm(v)
    if dist < d + 1.0:
        raise NotApplicable("goalkeeper is already next to the shooter")
    return _move_gk(shot, g + d * v / dist)


def set_body_part(shot: Shot, body: str) -> Shot:
    if body not in C.BODY_PART_GROUPS:
        raise ValueError(body)
    return replace(shot, body=body)


def mirror(shot: Shot) -> Shot:
    y, ff = mirror_y(shot.y, shot.ff)
    return replace(shot, y=float(y), ff=ff)


def place_shooter(shot: Shot, x: float, y: float) -> Shot:
    """Heat maps: the same situation with the shooter at (x, y)."""
    return replace(shot, x=float(x), y=float(y))


def keeper_only(gk_x: float = C.GOAL_X, gk_y: float = C.GOAL_CENTER[1]) -> np.ndarray:
    """Freeze frame with only the defending goalkeeper (on the goal line, centred by default)."""
    return np.array([[gk_x, gk_y, 0, 1]], np.float32)
