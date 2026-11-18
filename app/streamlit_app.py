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
def get_background() -> str:
    return B.png_data_url(B.pitch_background_png())


@st.cache_resource
def get_component():
    return st.components.v2.component(
        "xg_vision_board",
        js=(APP / "board.js").read_text(encoding="utf-8"),
        css=(APP / "board.css").read_text(encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_scenario(name: str):
    sc = get_scenarios()[name]
    ss = st.session_state
    ss.scenario = name
    ss.players = [dict(p) for p in sc["players"]]
    ss.body = sc["body"] if sc["body"] in BODY_LABELS else "foot"
    ss.version = ss.get("version", 0) + 1
    ss.last_key, ss.last_xg, ss.prev_xg = None, None, None


def on_scenario_change():
    load_scenario(st.session_state.scenario_select)


ss = st.session_state
scenarios = get_scenarios()
if "players" not in ss:                       # first run: ?scenario=<name> opens a given scenario
    start = st.query_params.get("scenario", "empty")
    load_scenario(start if start in scenarios else "empty")

# ---------------------------------------------------------------------------
# Sidebar: scenario, body part, mode
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Situation")
    names = list(scenarios)
    st.selectbox("Scenario", names, index=names.index(ss.scenario), key="scenario_select",
                 format_func=lambda n: scenarios[n]["label"], on_change=on_scenario_change)
    if st.button("Reset scenario", width="stretch"):
        load_scenario(ss.scenario)
    ss.body = st.radio("Body part of the shot", list(BODY_LABELS), index=list(BODY_LABELS).index(ss.body),
                       format_func=BODY_LABELS.get, horizontal=True)

    st.header("Board")
    mode = st.radio("Mode", ["Move", "Add"], horizontal=True,
                    help="Move: drag players. Add: click on the grass to add a player.")
    add_label = st.radio("Player to add", list(ADD_KINDS), horizontal=True, disabled=mode != "Add")
    st.caption("Double-click (or right-click) a player to remove it. The shooter (★) cannot be removed.")
    st.markdown("<span style='color:#ffd400'>★</span> shooter &nbsp; "
                "<span style='color:#1f77b4'>●</span> teammate &nbsp; "
                "<span style='color:#d62728'>●</span> defender &nbsp; ■ goalkeeper",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Board + panel
# ---------------------------------------------------------------------------
st.title("xG-Vision tactics board")
st.caption("A CNN reads a bird's-eye picture of the shot (player positions + body part) and outputs the "
           "probability of a goal. Trained on StatsBomb open data; see the "
           "[project README](https://github.com/Huseyinyagmur/xG-Vision).")

col_board, col_panel = st.columns([3, 2], gap="medium")
with col_board:
    result = get_component()(
        data={"bg": get_background(), "width": B.CANVAS_W, "height": B.CANVAS_H,
              "bounds": B.field_bounds_px(), "players": B.to_canvas(ss.players),
              "version": ss.version, "mode": mode.lower(), "add_kind": ADD_KINDS[add_label]},
        key="board", width="stretch", height="content", on_players_change=lambda: None,
    )
    state = getattr(result, "players", None)
    if state and state.get("version") == ss.version:          # ignore state from a previous scenario
        ss.players = B.from_canvas(state["items"])

with col_panel:
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

    st.metric("xG (CNN, mirror-averaged)", f"{pred.xg:.3f}",
              delta=None if delta is None else f"{delta:+.3f} last move",
              help="Mean of the prediction for this situation and for its mirror image (y → 80 − y).")
    st.caption(f"original {pred.original:.3f} · mirrored {pred.mirrored:.3f}")

    lr = float(get_logreg().predict(shot.x, shot.y, shot.body)[0])
    st.metric("Baseline: LogReg (distance + angle + body part)", f"{lr:.3f}",
              help="Only sees where the shooter is, not the other players.")
    st.caption(f"distance to goal centre {shot.distance:.1f} yd · goal-mouth angle "
               f"{np.degrees(shot.angle):.0f}°")

    for w in B.warnings_for(shot):
        st.warning(w, icon="⚠️")

    n = {k: sum(p["kind"] == k for p in ss.players) for k in B.KINDS}
    st.caption(f"{n['defender']} defenders · {n['goalkeeper']} goalkeeper · {n['teammate']} teammates")

    meta = scenarios[ss.scenario]["meta"]
    if meta:
        st.divider()
        st.markdown(f"**Original shot** ({meta['competition']} {meta['season']}): "
                    f"{meta['outcome']}, {meta['body_part'].lower()}, {meta['technique'].lower()} technique. "
                    f"StatsBomb xG {meta['statsbomb_xg']:.3f}.")
        st.caption("Valid for the untouched scenario only.")

st.caption("Data: StatsBomb Open Data. Model: SmallCNN (seed 42), step 2 of the project.")
