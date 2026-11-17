"""Shot edits used by the sanity tests (no data needed)."""
import numpy as np
import pytest

from src import config as C
from src import perturb as P

FF = np.array([[118.0, 41.0, 0, 1],      # defending goalkeeper
               [105.0, 35.0, 0, 0],      # defender
               [102.0, 50.0, 1, 0]],     # teammate
              np.float32)


def shot(x=100.0, y=30.0, body="foot", ff=FF):
    return P.Shot(x, y, ff.copy(), body)


@pytest.mark.parametrize("d", [3.0, -3.0])
def test_move_shooter_along_line(d):
    s = shot()
    m = P.move_shooter(s, d)
    assert np.isclose(np.hypot(m.x - s.x, m.y - s.y), abs(d))
    assert np.isclose(s.distance - m.distance, d)          # straight towards / away from the goal centre
    assert (m.ff == s.ff).all() and s.x == 100.0            # others fixed, input unchanged


def test_move_shooter_not_applicable():
    with pytest.raises(P.NotApplicable):
        P.move_shooter(shot(x=118.0, y=40.0), 3.0)          # 2 yd from goal
    with pytest.raises(P.NotApplicable):
        P.move_shooter(shot(x=61.0, y=40.0), -3.0)          # would leave the half


@pytest.mark.parametrize("xy", [(100.0, 30.0), (114.0, 20.0), (90.0, 40.0)])
def test_defenders_inside_and_outside_cone(xy):
    s = shot(*xy)
    for along in (1.5, s.distance / 2):
        p = P.defender_on_line(s, along)
        assert P.in_cone(*p, s.x, s.y)
    c = P.control_point(s, s.distance / 2, 6.0)
    assert not P.in_cone(*c, s.x, s.y)
    q = P.defender_on_line(s, s.distance / 2)
    assert np.isclose(np.hypot(*(c - q)), 6.0)
    # the new player is an outfield opponent -> channel 2 gains exactly one blob
    added = P.add_defender_on_line(s, 1.5)
    assert len(added.ff) == len(s.ff) + 1 and tuple(added.ff[-1, 2:]) == (0, 0)
    assert np.isclose(added.image()[2].sum() - s.image()[2].sum(), s.image()[0].sum(), rtol=1e-3)


def test_remove_goalkeeper_clears_channel():
    s = shot()
    assert s.image()[3].sum() > 0
    r = P.remove_goalkeeper(s)
    assert r.image()[3].sum() == 0 and r.gk_index() is None
    assert "no goalkeeper" in r.ood_reasons() and not s.ood_reasons()
    with pytest.raises(P.NotApplicable):
        P.remove_goalkeeper(r)


def test_shift_goalkeeper_sideways_moves_away_from_line():
    s = shot()
    _, n = s.line_dir()
    g0 = s.ff[s.gk_index(), :2]
    g1 = P.shift_goalkeeper_sideways(s, 3.0).ff[s.gk_index(), :2]
    off0, off1 = np.dot(g0 - s.pos, n), np.dot(g1 - s.pos, n)
    assert np.isclose(abs(off1) - abs(off0), 3.0, atol=1e-4)   # 3 yd further from the shot line
    assert np.isclose(np.hypot(*(g1 - g0)), 3.0, atol=1e-4)    # perpendicular move, length 3


def test_shift_goalkeeper_not_clipped_behind_goal_line():
    ff = FF.copy()
    ff[0, :2] = (119.5, 37.0)                 # keeper near the line; perpendicular move points past x = 120
    with pytest.raises(P.NotApplicable):
        P.shift_goalkeeper_sideways(shot(x=110.0, y=20.0, ff=ff), 3.0)


def test_shift_goalkeeper_forward():
    s = shot()
    g0 = s.ff[s.gk_index(), :2]
    g1 = P.shift_goalkeeper_forward(s, 3.0).ff[s.gk_index(), :2]
    assert np.isclose(np.hypot(*(s.pos - g0)) - np.hypot(*(s.pos - g1)), 3.0, atol=1e-4)


def test_body_part_changes_extra_only():
    s = shot()
    h = P.set_body_part(s, "head")
    assert (h.image() == s.image()).all()
    assert h.extra()[C.BODY_PART_GROUPS.index("head")] == 1 and h.extra().sum() == 1
    assert P.set_body_part(shot(x=90.0, y=40.0), "head").ood_reasons() == ["header from 25+ yd"]


def test_mirror_flips_rows():
    s = shot()
    np.testing.assert_allclose(P.mirror(s).image(), s.image()[:, ::-1], atol=1e-5)


def test_distance_angle_matches_data_columns():
    d, a = P.distance_angle(108.0, 40.0)
    assert np.isclose(d, 12.0) and np.isclose(a, 2 * np.arctan(4 / 12))
