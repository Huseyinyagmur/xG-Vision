"""Write the small files the tactics board needs, so the app runs without data/shots.pkl.

    python -m src.export_assets

app/assets/scenarios.json : empty pitch + 5 real test positions, chosen by rule (fixed seed);
                            real ones as (match_id, shot_id) only, fetched from StatsBomb at run time
app/assets/logreg.json    : LogReg (distance + angle + body part) scaler and coefficients
Model weights are not written here (models/small_cnn/best.pt locally; HF Hub later).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C
from src import perturb as P
from src.baselines import fit_logreg
from src.dataset import load_shots
from src.heatmap import pick_frames

EMPTY = {"name": "empty", "label": "Empty pitch: shooter 14 yd out, keeper on the line",
         "x": 106.0, "y": 40.0, "body": "foot", "freeze_frame": [[119.0, 40.0, 0, 1]], "meta": {}}


def _eligible(row) -> bool:
    s = P.Shot.from_row(row)
    k = s.gk_index()
    return k is not None and s.ff[k, 0] >= 114 and not s.ood_reasons() and s.body in ("foot", "head")


def pick_scenarios(test: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """crowded / counter as in the step-4 heat maps; tight angle, header, close range by rule."""
    frames = pick_frames(test)
    by_id = test.set_index("shot_id")
    picks = [("crowded", "Crowded box (real test shot)", by_id.loc[frames["crowded"][1]]),
             ("counter", "Counter attack, few defenders (real test shot)", by_id.loc[frames["counter"][1]])]
    rng = np.random.default_rng(C.BOARD_SCENARIO_SEED)
    rules = [
        ("tight_angle", "Tight angle (real test shot)",
         (test.x >= 110) & ((test.y - 40).abs() >= 10) & (test.body_part != "Head")),
        ("header", "Header from 6-12 yd (real test shot)",
         (test.body_part == "Head") & test.distance.between(6, 12)),
        ("close_range", "Close range, foot (real test shot)",
         (test.body_part != "Head") & (test.distance < 8)),
    ]
    for name, label, mask in rules:
        cand = test[mask]
        cand = cand[[_eligible(r) for r in cand.itertuples()]]
        row = cand.iloc[int(rng.integers(len(cand)))]
        picks.append((name, label, by_id.loc[row.shot_id]))
    return picks


def scenario_record(name: str, label: str, shot_id: str, row: pd.Series) -> dict:
    """Only identifiers: the positions are fetched from StatsBomb at run time (src.board),
    because the StatsBomb Public Data User Agreement (1.2.1) forbids redistributing the data."""
    return {"name": name, "label": label, "match_id": int(row.match_id), "shot_id": shot_id,
            "competition": row.competition, "season": row.season}


def logreg_record(train: pd.DataFrame) -> dict:
    model = fit_logreg(train, body_part=True)
    scaler, lr = model[0], model[-1]
    return {"model": "LogisticRegression on distance + angle + body part one-hot, fitted on the train split",
            "features": ["distance", "angle"] + [f"body_{g}" for g in C.BODY_PART_GROUPS],
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def main():
    C.APP_ASSETS.mkdir(parents=True, exist_ok=True)
    test = load_shots("test").reset_index(drop=True)
    scen = [EMPTY] + [scenario_record(n, l, r.name, r) for n, l, r in pick_scenarios(test)]
    (C.APP_ASSETS / "scenarios.json").write_text(json.dumps(scen, indent=1), encoding="utf-8")
    (C.APP_ASSETS / "logreg.json").write_text(json.dumps(logreg_record(load_shots("train")), indent=1),
                                              encoding="utf-8")
    for s in scen:
        print(f"{s['name']:12s} {s['label']:48s} {s.get('competition', '')} {s.get('season', '')}")
    print(f"-> {C.APP_ASSETS}")


if __name__ == "__main__":
    main()
