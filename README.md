# xG-Vision: Expected Goals from bird's-eye freeze frames

Classic expected-goals (xG) models use hand-crafted features such as distance and angle to goal.
**xG-Vision** asks whether a convolutional network can learn shot quality directly from a
picture of the situation: every shot is turned into a multi-channel bird's-eye "image" of the
attacking half (the shooter, teammates, defenders, goalkeeper and the shooting cone), built from
the player positions StatsBomb records at the moment of the shot (*freeze frames*).

> **Status:** step 1 (data pipeline and pitch tensors) is done. No model has been trained yet.

## Roadmap

| Step | Content | Status |
|---|---|---|
| 1 | Data pipeline: StatsBomb Open Data → cleaned shot table, match-level split, multi-channel pitch tensors, PyTorch `Dataset`, sanity plots | ✅ done |
| 2 | Baselines: logistic regression on distance/angle, StatsBomb xG as the reference | ⏳ planned |
| 3 | CNN on the pitch tensors (training, augmentation, tuning) | ⏳ planned |
| 4 | Evaluation and interpretation: log loss / Brier / ROC-AUC, calibration, what the network looks at | ⏳ planned |

## Data

[StatsBomb Open Data](https://github.com/statsbomb/open-data), all men's and women's
competitions with event data (snapshot downloaded November 2026).

| | |
|---|---|
| Matches | 3,961 |
| Open-play shots with a freeze frame | 95,160 |
| Goal rate | 10.4 % |
| Split (match-level, stratified by competition, 70/15/15) | train 66,754 · val 14,240 · test 14,166 shots |
| Reference: StatsBomb xG on the test split | log loss 0.278 · Brier 0.080 · ROC-AUC 0.807 |

Penalties are dropped. Free kicks and corners stay in the table but are excluded from
training. The split is done by match, so no match appears in more than one split.

## Pitch tensor

Each shot becomes a `(6, 160, 120)` tensor covering the attacking half
(x ∈ [60, 120], y ∈ [0, 80] StatsBomb yards, 0.5 yd per pixel). Tensors are rendered on the fly
in the `Dataset` (~1 ms per shot) instead of being stored. Mirroring the pitch left/right is
available as a training augmentation.

| ch | content |
|---|---|
| 0 | shooter (Gaussian blob, σ = 1 yd) |
| 1 | attacking teammates |
| 2 | defending outfield players |
| 3 | defending goalkeeper |
| 4 | shot cone: triangle from the shooter to both posts (binary) |
| 5 | static goal-distance map (CoordConv-style positional prior) |

![Freeze frame and the six tensor channels of one shot](results/plots/example_channels.png)

## Plots

**Data sanity check.** The observed goal rate falls with distance and closely follows the mean
StatsBomb xG. Shot and goal locations concentrate in the penalty box.

![Goal rate vs distance, shot locations and goals](results/plots/data_overview.png)

**Example freeze frames** from the training split (top row goals, bottom row misses).

![Example freeze frames](results/plots/freeze_frame_examples.png)

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/Huseyinyagmur/xG-Vision.git
cd xG-Vision
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate

# With an NVIDIA GPU, install the CUDA build of PyTorch first, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

```bash
python -m src.data_download                    # full dataset into data/ (~3,961 matches, cached per match)
python -m src.data_download --max-matches 50   # quick smoke test
python -m src.visualize                        # writes the plots to results/plots/
```

```python
from src.dataset import load_shots, ShotDataset

train = load_shots("train")                    # open-play shots of the training split
x, y, _ = ShotDataset(train, augment=True)[0]
x.shape                                        # torch.Size([6, 160, 120])
```

On Google Colab, put the project on Google Drive (see the comment in `src/config.py`) so the
download cache survives runtime resets.

## Project layout

```
src/
  config.py         # paths, pitch geometry, resolution, channels, split
  data_download.py  # StatsBomb -> shots + freeze frames -> clean table + match-level split
  pitch_tensor.py   # (shooter, freeze frame) -> (6, 160, 120) tensor
  dataset.py        # PyTorch Dataset, renders tensors on the fly (+ mirror augmentation)
  visualize.py      # pitch plots, channel plots, data overview
results/plots/      # figures shown above
data/               # downloaded data (not in the repo)
```

## Acknowledgements

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Pitch plots:
[mplsoccer](https://github.com/andrewRowlinson/mplsoccer).
