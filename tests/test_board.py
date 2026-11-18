"""Tactics board logic: coordinates, round trips, warnings, LogReg export (no Streamlit)."""
import json

import numpy as np
import pytest

from src import board as B
from src import config as C
from src import perturb as P
from src.pitch_tensor import render_shot

SCENARIOS = C.APP_ASSETS / "scenarios.json"
FF = np.array([[118.5, 41.0, 0, 1], [105.0, 35.0, 0, 0], [102.0, 50.0, 1, 0], [55.0, 30.0, 0, 0]], np.float32)


def test_pixel_mapping_orientation():
    # goal at the top, y = 0 on the left (the horizontal pitch turned 90° anticlockwise)
    assert B.to_px(C.X_MAX, C.Y_MIN) == (0.0, (C.BOARD_VIEW_X[1] - C.X_MAX) * C.BOARD_PX_PER_YD)
    assert B.to_px(C.X_MIN, C.Y_MAX) == (B.CANVAS_W, B.CANVAS_H)
    assert (B.CANVAS_W, B.CANVAS_H) == (640, 496)
    for x, y in [(100.0, 30.0), (119.9, 0.1), (60.0, 80.0)]:
        assert np.allclose(B.from_px(*B.to_px(x, y)), (x, y))
    assert B.from_px(-50, -50) == (C.X_MAX, C.Y_MIN)        # clamped onto the attacking half


def _roundtrip(shot: P.Shot) -> P.Shot:
    """Shot -> board players -> canvas items (JSON, as the browser sends them) -> shot."""
    items = json.loads(json.dumps(B.to_canvas(B.players_from_shot(shot))))
    return B.shot_from_players(B.from_canvas(items), shot.body)


def test_roundtrip_render_identical_synthetic():
    s = P.Shot(100.0, 30.0, FF, "foot")
    back = _roundtrip(s)
    np.testing.assert_allclose(back.image(), render_shot(s.x, s.y, s.ff), atol=1e-5)
    assert len(back.ff) == 3                                  # the player at x = 55 is outside the half


@pytest.mark.skipif(not SCENARIOS.exists(), reason="run `python -m src.export_assets` first")
def test_roundtrip_render_identical_real_scenarios():
    raw = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    assert len(raw) == 6 and raw[0]["name"] == "empty"
    for r in raw:
        ff = np.array(r["freeze_frame"], np.float32).reshape(-1, 4)
        s = P.Shot(r["x"], r["y"], ff, r["body"])
        back = _roundtrip(s)
        np.testing.assert_allclose(back.image(), render_shot(r["x"], r["y"], ff), atol=1e-5,
                                   err_msg=r["name"])
        assert back.body == r["body"]


def test_shot_from_players_errors():
    players = B.players_from_shot(P.Shot(100.0, 30.0, FF, "foot"))
    with pytest.raises(B.BoardError):
        B.shot_from_players([p for p in players if p["kind"] != "shooter"], "foot")
    with pytest.raises(B.BoardError):
        B.shot_from_players(players + [{"id": "g2", "kind": "goalkeeper", "x": 119.0, "y": 38.0}], "foot")


def test_warnings():
    assert B.warnings_for(P.Shot(100.0, 30.0, FF, "foot")) == []
    edge = B.warnings_for(P.Shot(119.0, 20.0, FF, "foot"))
    assert any("goal-line edge" in w for w in edge)
    assert not any("goal-line edge" in w for w in B.warnings_for(P.Shot(119.0, 41.0, FF, "foot")))
    no_gk = B.warnings_for(P.Shot(100.0, 30.0, FF[1:], "foot"))
    assert any("No goalkeeper" in w for w in no_gk)
    assert any("25+" in w for w in B.warnings_for(P.Shot(90.0, 40.0, FF, "head")))
    keeper_only = B.warnings_for(P.Shot(106.0, 40.0, FF[:1], "foot"))
    assert any("Only the goalkeeper" in w for w in keeper_only)


class _FakeModel:
    """Logit = 0.1 * (sum of the shooter channel's row index weights): differs from its mirror."""

    def __call__(self, x, e):
        import torch
        rows = torch.arange(x.shape[2], dtype=torch.float32)
        return (x[:, 0].sum(2) * rows).sum(1) / x[:, 0].sum((1, 2)) / 40 - 1


def test_tta_is_mean_of_original_and_mirror():
    import torch
    m = B.XGModel.__new__(B.XGModel)
    m.torch, m.device, m.model = torch, torch.device("cpu"), _FakeModel()
    s = P.Shot(100.0, 25.0, FF, "foot")
    pred = m.predict(s)
    assert pred.original != pytest.approx(pred.mirrored)
    assert pred.xg == pytest.approx((pred.original + pred.mirrored) / 2)


def test_logreg_json_matches_sklearn():
    pytest.importorskip("sklearn")
    if not C.SHOTS_PATH.exists():
        pytest.skip("needs data/shots.pkl")
    from src.baselines import design_matrix, fit_logreg
    from src.dataset import load_shots
    from src.export_assets import logreg_record
    train = load_shots("train")
    ref = fit_logreg(train, body_part=True)
    lr = B.LogRegJSON(json.loads(json.dumps(logreg_record(train))))
    test = load_shots("test").head(500)
    body = test.body_part.map(C.BODY_PART_MAP).to_numpy()
    np.testing.assert_allclose(lr.predict(test.x.to_numpy(), test.y.to_numpy(), body),
                               ref.predict_proba(design_matrix(test, True))[:, 1], rtol=1e-9)


def _fake_xgmodel():
    import torch
    m = B.XGModel.__new__(B.XGModel)
    m.torch, m.device, m.model = torch, torch.device("cpu"), _FakeModel()
    return m


def test_xg_map_grid_and_edge_mask():
    players = B.players_from_shot(P.Shot(100.0, 30.0, FF, "foot"))
    m = B.xg_map(_fake_xgmodel(), players, "foot", step=1.0, chunk=500)
    xs, ys = B.heat_axes(1.0)
    assert m["xg"].shape == (len(ys), len(xs)) == m["unreliable"].shape
    assert xs[0] == C.HEATMAP_X[0] and xs[-1] == C.HEATMAP_X[1] and ys[-1] == C.HEATMAP_Y[1]
    # the edge mask marks exactly x >= 118.5 outside the posts
    X, Y = np.meshgrid(xs, ys)
    np.testing.assert_array_equal(m["unreliable"], (X >= C.EDGE_WARN_X) & (np.abs(Y - 40) > 4))
    # the map does not depend on where the shooter was
    moved = [dict(p, x=90.0, y=60.0) if p["kind"] == "shooter" else p for p in players]
    np.testing.assert_allclose(B.xg_map(_fake_xgmodel(), moved, "foot", 1.0, 500)["xg"], m["xg"])


def test_overlay_cells_land_on_their_grid_points():
    xs, ys = B.heat_axes(2.0)
    xg = np.zeros((len(ys), len(xs)))
    xg[3, 5] = B.HEAT_VMAX                                     # one hot cell
    m = {"xs": xs, "ys": ys, "step": 2.0, "xg": xg, "unreliable": np.zeros_like(xg, bool)}
    img = B.overlay_rgba(m)
    assert img.shape == (B.CANVAS_H, B.CANVAS_W, 4)
    px, py = (int(v) for v in B.to_px(xs[5], ys[3]))
    hot = img[py, px, :3].astype(int).sum()
    cx, cy = (int(v) for v in B.to_px(xs[0], ys[0]))
    cold = img[cy, cx, :3].astype(int).sum()
    assert hot > cold + 300
    png = B.composite_png(B.pitch_background_png(), img)
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(png)).size == (B.CANVAS_W, B.CANVAS_H)
