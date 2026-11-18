"""xG-Vision tactics board (step 5, MVP).

    streamlit run app/streamlit_app.py

Drag players on a bird's-eye view of the attacking half and read the CNN's xG live.
xG is always the mean of the prediction for the situation and its mirror image (TTA).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                 # `streamlit run app/streamlit_app.py` from the project root
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from src import board as B
from src import config as C

st.set_page_config(page_title="xG-Vision tactics board", page_icon="⚽", layout="wide")

APP = Path(__file__).resolve().parent
BODY_LABELS = {"foot": "Foot", "head": "Head"}
ADD_KINDS = {"Defender": "defender", "Teammate": "teammate", "Goalkeeper": "goalkeeper"}
HEAT_STEPS = {"2 yd": 2.0, "1 yd (slow)": 1.0}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model ...")
def get_model() -> B.XGModel:
    return B.XGModel(C.MODELS_DIR / C.BOARD_MODEL / "best.pt")


@st.cache_resource
def get_logreg() -> B.LogRegJSON:
    return B.LogRegJSON.load(C.APP_ASSETS / "logreg.json")


@st.cache_resource
def get_scenarios() -> dict[str, dict]:
    return {s["name"]: s for s in B.load_scenarios(C.APP_ASSETS / "scenarios.json")}


@st.cache_resource
def get_background_png() -> bytes:
    return B.pitch_background_png()


@st.cache_resource
def get_component():
    return st.components.v2.component(
        "xg_vision_board",
        js=(APP / "board.js").read_text(encoding="utf-8"),
        css=(APP / "board.css").read_text(encoding="utf-8"),
    )


@st.cache_data(max_entries=64, show_spinner=False)
def get_xg_map(others: tuple, body: str, step: float) -> tuple[dict, str]:
    """xG map for a fixed set of non-shooter players (the shooter position does not matter:
    it is what the map varies), plus the board background with the map drawn on it."""
    players = [{"id": "s", "kind": "shooter", "x": 100.0, "y": 40.0}]
    players += [{"id": f"o{k}", "kind": kind, "x": x, "y": y} for k, (kind, x, y) in enumerate(others)]
    m = B.xg_map(get_model(), players, body, step)
    bg = B.png_data_url(B.composite_png(get_background_png(), B.overlay_rgba(m)))
    return m, bg



# ---------------------------------------------------------------------------
# State. Widgets with a key own their value in st.session_state; it is only changed in
# callbacks (before the widgets are drawn), never by passing a computed `index`.
# ---------------------------------------------------------------------------
DEFAULT_SCENARIO = "counter"


def load_scenario(name: str):
    sc = get_scenarios()[name]
    ss = st.session_state
    ss.scenario = ss.scenario_select = name
    ss.players = [dict(p) for p in sc["players"]]
    ss.body = sc["body"] if sc["body"] in BODY_LABELS else "foot"
    ss.version = ss.get("version", 0) + 1
    ss.last_key, ss.last_xg, ss.prev_xg = None, None, None


def on_scenario_change():
    load_scenario(st.session_state.scenario_select)


def on_reset():
    load_scenario(st.session_state.scenario)


def board_state() -> dict | None:
    """Last state the board component sent ({version, items}), or None."""
    s = st.session_state.get("board")
    if s is None:
        return None
    try:
        return s["players"]
    except (KeyError, TypeError):
        return getattr(s, "players", None)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


ss = st.session_state
scenarios = get_scenarios()
if "players" not in ss:                       # first run: ?scenario=<name> opens a given scenario
    start = st.query_params.get("scenario", DEFAULT_SCENARIO)
    load_scenario(start if start in scenarios else DEFAULT_SCENARIO)

# ---------------------------------------------------------------------------
# Sidebar: board mode and help
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Board")
    mode = st.radio("Mode", ["Move", "Add"], horizontal=True, key="mode",
                    help="Move: drag players. Add: click on the grass to add a player.")
    add_label = st.radio("Player to add", list(ADD_KINDS), horizontal=True, key="add_kind",
                         disabled=mode != "Add")
    st.caption("Double-click (or right-click) a player to remove it. The shooter (★) cannot be removed.")
    st.markdown("<span style='color:#ffd400'>★</span> shooter &nbsp; "
                "<span style='color:#1f77b4'>●</span> teammate &nbsp; "
                "<span style='color:#d62728'>●</span> defender &nbsp; ■ goalkeeper",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main: title, situation controls, board + panel
# ---------------------------------------------------------------------------
st.title("xG-Vision tactics board")
st.caption("A CNN reads a bird's-eye picture of the shot (player positions + body part) and outputs the "
           "probability of a goal. Trained on StatsBomb open data; see the "
           "[project README](https://github.com/Huseyinyagmur/xG-Vision).")

with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    st.selectbox("Scenario", list(scenarios), key="scenario_select", width=360,
                 format_func=lambda n: scenarios[n]["label"], on_change=on_scenario_change)
    st.button("Reset", on_click=on_reset, help="Back to the scenario as loaded")
    st.radio("Body part of the shot", list(BODY_LABELS), key="body", horizontal=True,
             format_func=BODY_LABELS.get)
    st.toggle("xG map", key="heat", help="Colour every spot by the xG a shot from there would get, "
              "with the other players where they are now.")
    st.radio("Map resolution", list(HEAT_STEPS), key="heat_res", horizontal=True,
             disabled=not ss.get("heat"))

# The component's last state is in session state under its key: take the players from there
# BEFORE drawing, so that the xG map below matches the board.
state = board_state()
if state and state.get("version") == ss.version:              # ignore state from a previous scenario
    ss.players = B.from_canvas(state["items"])

heat = None
background = B.png_data_url(get_background_png())
if ss.get("heat"):
    others = tuple(sorted((p["kind"], round(p["x"], 2), round(p["y"], 2))
                          for p in ss.players if p["kind"] != "shooter"))
    try:
        with st.spinner("Computing the xG map ..."):
            heat, background = get_xg_map(others, ss.body, HEAT_STEPS[ss.heat_res])
    except B.BoardError:
        heat = None

with st.container(horizontal=True, gap="medium", wrap=False, width=B.CANVAS_W + 340 + 32):
    with st.container(width="stretch"):
        get_component()(
            data={"bg": background, "width": B.CANVAS_W, "height": B.CANVAS_H,
                  "bounds": B.field_bounds_px(), "players": B.to_canvas(ss.players),
                  "version": ss.version, "mode": mode.lower(), "add_kind": ADD_KINDS[add_label]},
            key="board", width="stretch", height="content", on_players_change=lambda: None,
        )
        if heat is not None:
            i, j = np.unravel_index(np.where(heat["unreliable"], -1, heat["xg"]).argmax(), heat["xg"].shape)
            st.markdown(B.colorbar_html(), unsafe_allow_html=True)
            st.caption(f"Each cell: xG of a {BODY_LABELS[ss.body].lower()} shot from there, other players fixed "
                       f"({HEAT_STEPS[ss.heat_res]:g} yd grid). Best spot: x = {heat['xs'][j]:.0f}, "
                       f"y = {heat['ys'][i]:.0f} yd, xG {heat['xg'][i, j]:.2f}. Grey: goal-line edge (unreliable).")

    with st.container(width=340):
        try:
            shot = B.shot_from_players(ss.players, ss.body)
        except B.BoardError as err:
            st.error(f"Cannot compute xG: {err}.")
            st.stop()

        pred = get_model().predict(shot)
        key = (tuple((p["kind"], round(p["x"], 2), round(p["y"], 2)) for p in ss.players), ss.body)
        if key != ss.last_key:
            ss.prev_xg, ss.last_xg, ss.last_key = ss.last_xg, pred.xg, key
        delta = None if ss.prev_xg is None else pred.xg - ss.prev_xg
        body = BODY_LABELS[ss.body].lower()

        if B.on_goal_line_edge(shot):          # step 4: the model is not trustworthy here
            st.metric(f"xG (CNN, mirror-averaged) · {body} shot", "—",
                      help="Not shown: the model overrates shots from the goal-line edge (step 4).")
            st.caption(f"model output ~~{pred.xg:.3f}~~ (unreliable here)")
        else:
            st.metric(f"xG (CNN, mirror-averaged) · {body} shot", f"{pred.xg:.3f}",
                      delta=None if delta is None else f"{delta:+.3f} last move",
                      help="Mean of the prediction for this situation and for its mirror image (y → 80 − y).")
            st.caption(f"original {pred.original:.3f} · mirrored {pred.mirrored:.3f}")

        lr = float(get_logreg().predict(shot.x, shot.y, shot.body)[0])
        st.metric(f"Baseline: LogReg (distance + angle + body part) · {body}", f"{lr:.3f}",
                  help="Only sees where the shooter is, not the other players.")
        st.caption(f"distance to goal centre {shot.distance:.1f} yd · goal-mouth angle "
                   f"{np.degrees(shot.angle):.0f}°")

        for w in B.warnings_for(shot):
            st.warning(w, icon="⚠️")
        note = B.keeper_note(shot)
        if note:
            st.info(note, icon="ℹ️")

        n = {k: sum(p["kind"] == k for p in ss.players) for k in B.KINDS}
        st.caption(f"{plural(n['defender'], 'defender')} · {plural(n['goalkeeper'], 'goalkeeper')} · "
                   f"{plural(n['teammate'], 'teammate')}")

        meta = scenarios[ss.scenario]["meta"]
        if meta:
            st.divider()
            st.markdown(f"**Original shot** ({meta['competition']} {meta['season']}): "
                        f"{meta['outcome']}, {meta['body_part'].lower()}, {meta['technique'].lower()} technique. "
                        f"StatsBomb xG {meta['statsbomb_xg']:.3f}.")
            st.caption("Valid for the untouched scenario only.")

st.caption("Data: StatsBomb Open Data. Model: SmallCNN (seed 42), step 2 of the project.")
