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
def on_goal_line_edge(shot: P.Shot) -> bool:
    """Shooter within 1.5 yd of the goal line and outside the posts (step 4 edge artefact)."""
    return shot.x >= C.EDGE_WARN_X and abs(shot.y - C.GOAL_CENTER[1]) > POST_HALF


def warnings_for(shot: P.Shot) -> list[str]:
    """Situations where the model should not be trusted."""
    w = []
    if on_goal_line_edge(shot):
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


KEEPER_NOTE_YD = 3.0


def keeper_note(shot: P.Shot) -> str | None:
    """Step 5 finding: the model's response to the keeper's position is confounded."""
    k = shot.gk_index()
    if k is None or C.GOAL_X - shot.ff[k, 0] <= KEEPER_NOTE_YD:
        return None
    return (f"The goalkeeper is {C.GOAL_X - shot.ff[k, 0]:.0f} yd off the line. In the data a keeper off the "
            "line often means a keeper out of position, so the model raises xG when the keeper steps forward; "
            "it has not learned the effect of narrowing the angle.")


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
# xG map layer: the shooter moved over a grid, everyone else fixed
# ---------------------------------------------------------------------------
HEAT_VMAX = 0.6
HEAT_TICKS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6)


def heat_norm(v):
    """Square-root colour scale: most xG values are small, so 0.01-0.1 needs room on the scale."""
    return np.sqrt(np.clip(np.asarray(v, np.float64) / HEAT_VMAX, 0.0, 1.0))


def heat_axes(step: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(C.HEATMAP_X[0], C.HEATMAP_X[1] + 1e-9, step)
    ys = np.arange(C.HEATMAP_Y[0], C.HEATMAP_Y[1] + 1e-9, step)
    return xs, ys


def xg_map(model: XGModel, players: list[dict], body: str, step: float = 2.0, chunk: int = 128) -> dict:
    """Mirror-averaged xG with the shooter at every grid point (the other players stay put).
    Returns {xs, ys, step, xg (len(ys), len(xs)), unreliable (same shape, goal-line edge)}."""
    base = shot_from_players(players, body)
    xs, ys = heat_axes(step)
    shots = [P.place_shooter(base, x, y) for y in ys for x in xs]
    xg = []
    for i in range(0, len(shots), chunk):            # chunks keep memory small (2 images per shot)
        xg += [p.xg for p in model.predict_many(shots[i:i + chunk])]
    shape = (len(ys), len(xs))
    return {"xs": xs, "ys": ys, "step": step, "xg": np.array(xg).reshape(shape),
            "unreliable": np.array([on_goal_line_edge(s) for s in shots]).reshape(shape)}


def heat_colormap():
    import matplotlib
    return matplotlib.colormaps["magma"]


def overlay_rgba(m: dict, alpha: tuple[float, float] = (0.12, 0.78)) -> np.ndarray:
    """(CANVAS_H, CANVAS_W, 4) uint8 image: one square cell per grid point, colour = xG on a
    square-root scale, opacity rising with xG so that the pitch stays visible where xG is low;
    unreliable cells (goal-line edge) are grey."""
    cmap = heat_colormap()
    img = np.zeros((CANVAS_H, CANVAS_W, 4), np.uint8)
    half = m["step"] * PPY / 2
    for i, y in enumerate(m["ys"]):
        for j, x in enumerate(m["xs"]):
            px, py = to_px(x, y)
            r0, r1 = int(round(py - half)), int(round(py + half))
            c0, c1 = int(round(px - half)), int(round(px + half))
            r0, c0 = max(r0, 0), max(c0, 0)
            if m["unreliable"][i, j]:
                rgba = (150, 150, 150, int(255 * alpha[1]))
            else:
                v = float(heat_norm(m["xg"][i, j]))
                r, g, b, _ = cmap(0.15 + 0.85 * v)          # skip the near-black start of magma
                rgba = (int(255 * r), int(255 * g), int(255 * b), int(255 * (alpha[0] + (alpha[1] - alpha[0]) * v)))
            img[r0:r1, c0:c1] = rgba
    return img


def composite_png(background_png: bytes, overlay: np.ndarray) -> bytes:
    from PIL import Image
    bg = Image.open(io.BytesIO(background_png)).convert("RGBA")
    out = Image.alpha_composite(bg, Image.fromarray(overlay, "RGBA"))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def colorbar_html(width: int = 260) -> str:
    """HTML legend of the map colours with tick labels at their square-root positions."""
    cmap = heat_colormap()
    stops = []
    for k in range(11):
        r, g, b, _ = cmap(0.15 + 0.85 * k / 10)
        stops.append(f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)}) {10 * k}%")
    ticks = "".join(
        f"<span style='position:absolute;left:{100 * float(heat_norm(v)):.1f}%;transform:translateX(-50%)'>"
        f"{v:g}{'+' if v == HEAT_VMAX else ''}</span>" for v in HEAT_TICKS)
    return (f"<div style='font-size:0.75rem;color:#666;width:{width}px;margin:4px 0 18px 8px'>"
            f"<div style='height:10px;border-radius:2px;background:linear-gradient(to right, {', '.join(stops)})'></div>"
            f"<div style='position:relative;height:1em'>{ticks}</div></div>")


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
def shot_from_event(rec: dict) -> P.Shot:
    """One shot as extracted by data_download (raw StatsBomb event) -> Shot, with the same
    clipping as data_download.clean_shots, so the tensor equals the one used in evaluation."""
    ff = np.clip(np.asarray(rec["freeze_frame"], np.float32).reshape(-1, 4), [0, 0, 0, 0],
                 [C.PITCH_LENGTH, C.PITCH_WIDTH, 1, 1])
    x = float(np.clip(rec["x"], 0, C.PITCH_LENGTH))
    y = float(np.clip(rec["y"], 0, C.PITCH_WIDTH))
    return P.Shot(x, y, ff, C.BODY_PART_MAP[rec["body_part"]])


def fetch_event_shot(match_id: int, shot_id: str) -> dict:
    """The shot's event record, downloaded from StatsBomb's own open-data repository (and
    cached under data/cache/), so this project does not redistribute the raw data."""
    from src.data_download import fetch_match_shots
    for rec in fetch_match_shots(int(match_id)):
        if rec["shot_id"] == shot_id:
            return rec
    raise KeyError(f"shot {shot_id} not found in match {match_id}")


def load_scenarios(path: str | Path = C.APP_ASSETS / "scenarios.json", fetch=fetch_event_shot
                   ) -> tuple[list[dict], list[str]]:
    """Scenarios [{name, label, body, players, meta}] (players in yards) and a list of errors.
    Synthetic scenarios carry their positions; real ones only (match_id, shot_id) and are
    fetched with `fetch`. A real scenario that cannot be fetched (offline) is skipped."""
    out, errors = [], []
    for s in json.loads(Path(path).read_text(encoding="utf-8")):
        if "freeze_frame" in s:                                   # synthetic
            shot = P.Shot(s["x"], s["y"], np.array(s["freeze_frame"], np.float32).reshape(-1, 4), s["body"])
            meta = {}
        else:
            try:
                rec = fetch(s["match_id"], s["shot_id"])
            except Exception as err:                              # network, missing shot ...
                errors.append(f"{s['label']}: {type(err).__name__}: {err}")
                continue
            shot = shot_from_event(rec)
            meta = {"shot_id": s["shot_id"], "competition": s["competition"], "season": s["season"],
                    "outcome": rec["outcome"], "statsbomb_xg": float(rec["statsbomb_xg"]),
                    "body_part": rec["body_part"], "technique": rec["technique"]}
        out.append({"name": s["name"], "label": s["label"], "body": shot.body,
                    "players": players_from_shot(shot), "meta": meta})
    return out, errors


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
