"""Logic of the interactive tactics board (step 5), independent of Streamlit.

Coordinates: players are stored in StatsBomb yards. The board shows the attacking half
vertically, goal at the top (the horizontal pitch turned 90° anticlockwise, so y = 0 is on the
left, as in mplsoccer's VerticalPitch):

    px = (y - 0) * BOARD_PX_PER_YD          py = (BOARD_VIEW_X[1] - x) * BOARD_PX_PER_YD

Players are dicts {"id", "kind", "x", "y"} with kind in KINDS. Exactly one shooter is required.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config as C
from src import perturb as P

KINDS = ("shooter", "teammate", "defender", "goalkeeper")
PPY = C.BOARD_PX_PER_YD
VIEW_X_LO, VIEW_X_HI = C.BOARD_VIEW_X
CANVAS_W = int(round((C.Y_MAX - C.Y_MIN) * PPY))            # 640
CANVAS_H = int(round((VIEW_X_HI - VIEW_X_LO) * PPY))        # 496
POST_HALF = (C.POST_HIGH_Y - C.POST_LOW_Y) / 2              # 4 yd


class BoardError(ValueError):
    """The situation on the board cannot be scored (e.g. no shooter)."""


# ---------------------------------------------------------------------------
# Coordinates and conversions
# ---------------------------------------------------------------------------
def to_px(x: float, y: float) -> tuple[float, float]:
    return (y - C.Y_MIN) * PPY, (VIEW_X_HI - x) * PPY


def from_px(px: float, py: float) -> tuple[float, float]:
    """Canvas pixel -> StatsBomb yards, clamped to the attacking half."""
    x = VIEW_X_HI - py / PPY
    y = C.Y_MIN + px / PPY
    return float(np.clip(x, C.X_MIN, C.X_MAX)), float(np.clip(y, C.Y_MIN, C.Y_MAX))


def field_bounds_px() -> dict:
    """Where players may be placed on the canvas (the attacking half, not behind the goal line)."""
    x0, y0 = to_px(C.X_MAX, C.Y_MIN)
    x1, y1 = to_px(C.X_MIN, C.Y_MAX)
    return {"left": x0, "top": y0, "right": x1, "bottom": y1}


def players_from_shot(shot: P.Shot) -> list[dict]:
    """Shooter + freeze frame -> board players. Players outside the rendered half are dropped
    (render_shot ignores them too). An attacking goalkeeper is an ordinary teammate for the
    model (channel 1), so it becomes a teammate."""
    out = [{"id": "s", "kind": "shooter", "x": float(shot.x), "y": float(shot.y)}]
    for k, (x, y, team, gk) in enumerate(np.asarray(shot.ff, np.float64)):
        if not C.X_MIN <= x <= C.X_MAX:
            continue
        kind = "teammate" if team == 1 else ("goalkeeper" if gk == 1 else "defender")
        out.append({"id": f"p{k}", "kind": kind, "x": float(x), "y": float(y)})
    return out


def shot_from_players(players: list[dict], body: str) -> P.Shot:
    shooters = [p for p in players if p["kind"] == "shooter"]
    if len(shooters) != 1:
        raise BoardError(f"need exactly one shooter, found {len(shooters)}")
    if sum(p["kind"] == "goalkeeper" for p in players) > 1:
        raise BoardError("more than one goalkeeper")
    flags = {"teammate": (1, 0), "defender": (0, 0), "goalkeeper": (0, 1)}
    rows = [[p["x"], p["y"], *flags[p["kind"]]] for p in players if p["kind"] != "shooter"]
    ff = np.array(rows, np.float32).reshape(-1, 4)
    return P.Shot(float(shooters[0]["x"]), float(shooters[0]["y"]), ff, body)


def to_canvas(players: list[dict]) -> list[dict]:
    out = []
    for p in players:
        px, py = to_px(p["x"], p["y"])
        out.append({"id": p["id"], "kind": p["kind"], "px": px, "py": py})
    return out


def from_canvas(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        if it.get("kind") not in KINDS:
            continue
        x, y = from_px(float(it["px"]), float(it["py"]))
        out.append({"id": str(it["id"]), "kind": it["kind"], "x": x, "y": y})
    return out


# ---------------------------------------------------------------------------
# Warnings (step 4 findings)
# ---------------------------------------------------------------------------
def warnings_for(shot: P.Shot) -> list[str]:
    """Situations where the model should not be trusted."""
    w = []
    if shot.x >= C.EDGE_WARN_X and abs(shot.y - C.GOAL_CENTER[1]) > POST_HALF:
        w.append("Shooter is on the goal-line edge outside the posts: the model overrates such shots "
                 "(step 4 found xG ≈ 0.45 even 28 yd wide). Do not trust this value.")
    if not (shot.ff[:, 3] == 0).any():
        w.append("Only the goalkeeper on the pitch: out of distribution (never happens in training). "
                 "Treat the value as illustrative.")
    for reason in shot.ood_reasons():
        if reason == "no goalkeeper":
            w.append("No goalkeeper: out of distribution (0.1% of the training shots).")
        else:
            w.append(f"Out of distribution: {reason} (almost absent from the training data).")
    return w


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    xg: float           # test-time augmentation: mean of the two below
    original: float
    mirrored: float


class XGModel:
    """The CNN with mirror test-time augmentation."""

    def __init__(self, checkpoint: str | Path, device=None):
        import torch
        from src.models import load_checkpoint
        from src.utils import get_device
        self.torch = torch
        self.device = device or get_device()
        self.model, _ = load_checkpoint(checkpoint, self.device)

    def predict(self, shot: P.Shot) -> Prediction:
        return self.predict_many([shot])[0]

    def predict_many(self, shots: list[P.Shot]) -> list[Prediction]:
        torch = self.torch
        both = [s for shot in shots for s in (shot, P.mirror(shot))]
        x = torch.from_numpy(np.stack([s.image() for s in both])).to(self.device)
        e = torch.from_numpy(np.stack([s.extra() for s in both])).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x, e).float().reshape(-1)).cpu().numpy().astype(float)
        return [Prediction(xg=(a + b) / 2, original=a, mirrored=b) for a, b in zip(p[0::2], p[1::2])]


class LogRegJSON:
    """LogReg (distance + angle + body part) from exported coefficients (no sklearn, no data)."""

    def __init__(self, params: dict):
        self.mean = np.asarray(params["scaler_mean"], np.float64)
        self.scale = np.asarray(params["scaler_scale"], np.float64)
        self.coef = np.asarray(params["coef"], np.float64)
        self.intercept = float(params["intercept"])
        assert params["features"][:2] == ["distance", "angle"]

    @classmethod
    def load(cls, path: str | Path) -> "LogRegJSON":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, x, y, body) -> np.ndarray:
        d, a = P.distance_angle(x, y)
        d, a = np.atleast_1d(d), np.atleast_1d(a)
        bodies = np.broadcast_to(np.asarray(body), d.shape)
        onehot = np.array([[b == g for g in C.BODY_PART_GROUPS] for b in bodies], np.float64)
        z = ((np.column_stack([d, a, onehot]) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
def load_scenarios(path: str | Path = C.APP_ASSETS / "scenarios.json") -> list[dict]:
    """[{name, label, body, players, meta}] with players in yards."""
    out = []
    for s in json.loads(Path(path).read_text(encoding="utf-8")):
        shot = P.Shot(s["x"], s["y"], np.array(s["freeze_frame"], np.float32).reshape(-1, 4), s["body"])
        out.append({"name": s["name"], "label": s["label"], "body": s["body"],
                    "players": players_from_shot(shot), "meta": s.get("meta", {})})
    return out


def pitch_background_png() -> bytes:
    """Vertical attacking half as a PNG of exactly CANVAS_W x CANVAS_H pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mplsoccer import VerticalPitch

    dpi = 100
    fig = plt.figure(figsize=(CANVAS_W / dpi, CANVAS_H / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, pitch_color="#3d8c40", line_color="white",
                          linewidth=1.6, goal_type="box", pad_top=2, pad_bottom=0, pad_left=0, pad_right=0)
    pitch.draw(ax=ax)
    ax.set_xlim(C.Y_MIN, C.Y_MAX)               # VerticalPitch: data x = StatsBomb y (0 on the left)
    ax.set_ylim(VIEW_X_LO, VIEW_X_HI)           # data y = StatsBomb x (goal at the top)
    ax.set_facecolor("#3d8c40")
    fig.patch.set_facecolor("#3d8c40")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
