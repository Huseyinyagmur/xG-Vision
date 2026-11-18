"""Download StatsBomb open data, extract shots + freeze frames, build a clean shot table.

Usage (Colab / terminal):
    python -m src.data_download                 # all competitions
    python -m src.data_download --max-matches 50  # quick smoke test

Only the shots of each match are cached (a few KB per match instead of ~4 MB of
raw events), so re-running is cheap.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from src import config as C

_session = requests.Session()


def _get_json(url: str):
    r = _session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 1) Match list
# ---------------------------------------------------------------------------
def list_matches(genders=C.GENDERS) -> pd.DataFrame:
    comps = _get_json(f"{C.STATSBOMB_RAW}/competitions.json")
    comps = [c for c in comps if c["competition_gender"] in genders and c["match_available"]]
    rows = []
    for c in tqdm(comps, desc="competitions"):
        url = f"{C.STATSBOMB_RAW}/matches/{c['competition_id']}/{c['season_id']}.json"
        for m in _get_json(url):
            rows.append({
                "match_id": m["match_id"],
                "competition": c["competition_name"],
                "season": c["season_name"],
                "gender": c["competition_gender"],
                "match_date": m.get("match_date"),
            })
    return pd.DataFrame(rows).drop_duplicates("match_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) Shots of one match (cached)
# ---------------------------------------------------------------------------
def _extract_shots(events: list[dict]) -> list[dict]:
    shots = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        s = e["shot"]
        players = []
        for p in s.get("freeze_frame", []) or []:
            is_gk = p["position"]["name"] == "Goalkeeper"
            players.append([p["location"][0], p["location"][1], int(p["teammate"]), int(is_gk)])
        shots.append({
            "shot_id": e["id"],
            "period": e["period"],
            "minute": e["minute"],
            "team": e["team"]["name"],
            "player": e["player"]["name"],
            "x": e["location"][0],
            "y": e["location"][1],
            "shot_type": s["type"]["name"],
            "body_part": s["body_part"]["name"],
            "technique": s["technique"]["name"],
            "outcome": s["outcome"]["name"],
            "statsbomb_xg": s.get("statsbomb_xg"),
            "first_time": bool(s.get("first_time", False)),
            "one_on_one": bool(s.get("one_on_one", False)),
            "under_pressure": bool(e.get("under_pressure", False)),
            "freeze_frame": players,   # list of [x, y, teammate, is_gk]
        })
    return shots


def fetch_match_shots(match_id: int) -> list[dict]:
    path = C.CACHE_DIR / f"{match_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    shots = _extract_shots(_get_json(f"{C.STATSBOMB_RAW}/events/{match_id}.json"))
    path.write_text(json.dumps(shots))
    return shots


# ---------------------------------------------------------------------------
# 3) Cleaning
# ---------------------------------------------------------------------------
def clean_shots(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df["n_players"] = df["freeze_frame"].map(len)
    df["has_gk"] = df["freeze_frame"].map(lambda ff: any(p[2] == 0 and p[3] == 1 for p in ff))
    df["is_goal"] = (df["outcome"] == "Goal").astype(np.int8)

    # Penalties have a degenerate freeze frame (everyone outside the box) -> drop.
    # Other set pieces are kept in the table and filtered via TRAIN_SHOT_TYPES later.
    df = df[df["shot_type"] != "Penalty"]
    df = df[df["n_players"] > 0]                     # no freeze frame -> unusable
    df = df[df["x"] >= C.X_MIN]                      # shots from own half: rare, out of image
    df = df[df["statsbomb_xg"].notna()]

    # Freeze-frame coordinates as float arrays, players clipped to the pitch
    df["freeze_frame"] = df["freeze_frame"].map(
        lambda ff: np.clip(np.asarray(ff, dtype=np.float32), [0, 0, 0, 0], [C.PITCH_LENGTH, C.PITCH_WIDTH, 1, 1])
    )
    df["x"] = df["x"].clip(0, C.PITCH_LENGTH)
    df["y"] = df["y"].clip(0, C.PITCH_WIDTH)

    # Classic hand-crafted features (used by the logistic-regression baseline)
    dx, dy = C.GOAL_X - df["x"], C.GOAL_CENTER[1] - df["y"]
    df["distance"] = np.hypot(dx, dy)
    a1 = np.arctan2(C.POST_LOW_Y - df["y"], dx)
    a2 = np.arctan2(C.POST_HIGH_Y - df["y"], dx)
    df["angle"] = np.abs(a2 - a1)                    # opening angle to the goal mouth (rad)

    print(f"clean_shots: {n0} -> {len(df)} shots")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) Match-level split (stratified by competition so every league is in every split)
# ---------------------------------------------------------------------------
def assign_match_split(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fr = C.SPLIT_FRACTIONS
    split_of = {}
    for _, g in df.groupby("competition"):
        ids = np.array(sorted(g["match_id"].unique()))
        rng.shuffle(ids)
        n_tr = int(round(fr["train"] * len(ids)))
        n_va = int(round(fr["val"] * len(ids)))
        for i, mid in enumerate(ids):
            split_of[mid] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    df = df.copy()
    df["split"] = df["match_id"].map(split_of)
    # sanity: no match appears in more than one split
    assert df.groupby("match_id")["split"].nunique().max() == 1
    return df


def training_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Shots used for model training/evaluation (open play by default)."""
    return df[df["shot_type"].isin(C.TRAIN_SHOT_TYPES)].reset_index(drop=True)


# ---------------------------------------------------------------------------
def build_dataset(max_matches: int | None = None) -> pd.DataFrame:
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matches = list_matches()
    if max_matches:
        matches = matches.sample(max_matches, random_state=C.SEED)
    print(f"{len(matches)} matches")

    rows = []
    with ThreadPoolExecutor(C.DOWNLOAD_WORKERS) as pool:
        futs = {pool.submit(fetch_match_shots, mid): mid for mid in matches["match_id"]}
        for f in tqdm(as_completed(futs), total=len(futs), desc="matches"):
            mid = futs[f]
            try:
                for s in f.result():
                    s["match_id"] = mid
                    rows.append(s)
            except Exception as ex:  # keep going if a single file fails
                print(f"match {mid} failed: {ex}")

    df = pd.DataFrame(rows).merge(matches, on="match_id", how="left")
    df = assign_match_split(clean_shots(df))
    C.SHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(C.SHOTS_PATH)
    print(f"saved {len(df)} shots -> {C.SHOTS_PATH}")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\nShots by type:\n", df.groupby("shot_type")["is_goal"].agg(["size", "mean"]).round(3))
    tr = training_subset(df)
    print(f"\nTraining subset ({', '.join(C.TRAIN_SHOT_TYPES)}): {len(tr)} shots, "
          f"goal rate {tr['is_goal'].mean():.3f}, matches {tr['match_id'].nunique()}")
    print(tr.groupby("split").agg(shots=("is_goal", "size"), goal_rate=("is_goal", "mean"),
                                  matches=("match_id", "nunique")).round(3))
    print("\nGender:\n", tr["gender"].value_counts())
    print("\nBody part:\n", tr.groupby("body_part")["is_goal"].agg(["size", "mean"]).round(3))
    print(f"\nShots without goalkeeper in freeze frame: {(~tr['has_gk']).mean():.2%}")
    print(f"Players per freeze frame: median {tr['n_players'].median():.0f}, "
          f"min {tr['n_players'].min()}, max {tr['n_players'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-matches", type=int, default=None)
    args = ap.parse_args()
    summarize(build_dataset(args.max_matches))
