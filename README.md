# xG-Vision: Expected Goals from bird's-eye freeze frames

Classic expected-goals (xG) models use hand-crafted features such as distance and angle to goal.
**xG-Vision** asks whether a convolutional network can learn shot quality directly from a
picture of the situation: every shot is turned into a multi-channel bird's-eye "image" of the
attacking half (the shooter, teammates, defenders, goalkeeper and the shooting cone), built from
the player positions StatsBomb records at the moment of the shot (*freeze frames*).

> **Status:** steps 1–2 are done: data pipeline, and two CNNs (a small one trained from
> scratch and an ImageNet ResNet18) trained with early stopping. Test-set evaluation is step 3.

## Roadmap

| Step | Content | Status |
|---|---|---|
| 1 | Data pipeline: StatsBomb Open Data → cleaned shot table, match-level split, multi-channel pitch tensors, PyTorch `Dataset`, sanity plots | ✅ done |
| 2 | Models: small CNN from scratch vs. ImageNet ResNet18 (6-channel first conv), body part as an extra input, training loop with early stopping | ✅ done |
| 3 | Evaluation on the test split: log loss / Brier / ROC-AUC, calibration, comparison with StatsBomb xG and a distance + angle logistic regression, interesting shots, Grad-CAM | ⏳ next |
| 4 | Sanity tests: move the shooter, add a defender in the shooting lane, remove the goalkeeper; xG heat map over the box | ⏳ planned |
| 5 | Interactive tactics board (Streamlit or Gradio): drag players, get xG live | ⏳ planned |
| 6 | (optional) Broadcast frame → player detection + pitch homography → bird's-eye view → xG | ⏳ planned |

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

Penalties are dropped. Free kicks (and a handful of corner and kick-off shots) stay in the
table but are excluded from training. The split is done by match, so no match appears in more
than one split.

## Pitch tensor

Each shot becomes a `(6, 160, 120)` tensor covering the attacking half
(x ∈ [60, 120], y ∈ [0, 80] StatsBomb yards, 0.5 yd per pixel). Tensors are rendered on the fly
in the `Dataset` (~1.2 ms per shot on one CPU core; each player's Gaussian is only evaluated in a
small window around it) instead of being stored. Mirroring the pitch left/right is used as a
training augmentation; a test checks that it equals flipping the tensor rows.

| ch | content |
|---|---|
| 0 | shooter (Gaussian blob, σ = 1 yd) |
| 1 | attacking teammates |
| 2 | defending outfield players |
| 3 | defending goalkeeper |
| 4 | shot cone: triangle from the shooter to both posts (binary) |
| 5 | static goal-distance map (CoordConv-style positional prior) |

![Freeze frame and the six tensor channels of one shot](results/plots/example_channels.png)

## Models

Both models share the interface `forward(image, extra) -> logit`: the CNN turns the pitch
tensor into a feature vector (global average pooling), the body part (head / foot / other,
one-hot) is concatenated, and a small MLP head outputs the goal logit. Left and right foot are
merged so that the mirror augmentation stays valid.

| | SmallCNN | ResNet18 |
|---|---|---|
| Backbone | 5 × [conv-BN-ReLU × 2 + max-pool], 16-32-64-128-256 channels, from scratch | torchvision ResNet18, ImageNet weights; first conv rebuilt for 6 channels (mean RGB filter, rescaled) |
| Parameters | 1.2 M | 11.2 M |
| Learning rate | 1e-3 | 1e-3 head and first conv, 1e-4 pretrained layers |

Training: plain binary cross-entropy (no class weights, to keep probabilities calibrated),
AdamW, LR halved on a val plateau, bf16 mixed precision, batch 256, early stopping on the
validation log loss (patience 5). Before every validation the BatchNorm statistics are
recomputed with the frozen weights ("precise BN"): the running averages lag behind the weights
and had pushed the mean predicted xG to 0.03–0.05 instead of the true goal rate of 0.10.

## Results so far (validation split)

| Model | Log loss | Brier | ROC-AUC | Mean xG | Best epoch |
|---|---|---|---|---|---|
| SmallCNN | **0.2656** | **0.0765** | **0.811** | 0.109 | 9 / 14 |
| ResNet18 (ImageNet) | 0.2690 | 0.0772 | 0.806 | 0.099 | 5 / 10 |
| StatsBomb xG | 0.2629 | 0.0753 | 0.814 | 0.100 | – |
| Constant (training goal rate) | 0.3323 | 0.0926 | 0.500 | 0.103 | – |

Validation goal rate: 0.103 (14,240 shots). By gender, SmallCNN reaches log loss 0.2617 /
AUC 0.819 on men's and 0.2728 / 0.796 on women's matches. From the pitch picture plus the body
part alone, the small CNN gets close to StatsBomb's xG model. These numbers slightly favour
the CNNs, because the epoch was selected on this split; the real comparison is on the test
split (step 3).

The pretrained ResNet18 is not better than the small CNN and starts to overfit after epoch 5,
while the small CNN improves steadily:

![SmallCNN training curves](results/plots/training_curves_small_cnn.png)
![ResNet18 training curves](results/plots/training_curves_resnet18.png)

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
python -m src.evaluate                         # reference metrics (StatsBomb xG, constant) on val/test

python -m src.train --model small_cnn          # ~11 min on an RTX 4050 Laptop GPU
python -m src.train --model resnet18           # ~8 min
python -m src.train --model small_cnn --subset 15000 --epochs 3   # quick smoke test

python -m pytest tests -q                      # tests needing data/shots.pkl are skipped without it
```

Each run writes `models/<run>/` (`best.pt`, `last.pt`, `history.csv`, `train.log`,
`val_predictions.csv`) and `results/plots/training_curves_<run>.png`.

```python
from src.dataset import load_shots, ShotDataset
from src.models import load_checkpoint

train = load_shots("train")                    # open-play shots of the training split
x, extra, y, _ = ShotDataset(train, augment=True)[0]
x.shape, extra                                 # torch.Size([6, 160, 120]), body part one-hot

model, ckpt = load_checkpoint("models/small_cnn/best.pt")
xg = model(x[None], extra[None]).sigmoid()     # P(goal)
```

On Google Colab, put the project on Google Drive (see the comment in `src/config.py`) so the
download cache survives runtime resets.

## Project layout

```
src/
  config.py         # paths, pitch geometry, resolution, channels, split
  data_download.py  # StatsBomb -> shots + freeze frames -> clean table + match-level split
  pitch_tensor.py   # (shooter, freeze frame) -> (6, 160, 120) tensor
  dataset.py        # PyTorch Dataset (renders on the fly, mirror augmentation) + phase batch sampler
  features.py       # scalar model inputs (body part one-hot)
  models.py         # SmallCNN, ResNet18xG, shared head, checkpoint loading
  train.py          # training loop, precise BN, early stopping, checkpoints
  evaluate.py       # log loss / Brier / ROC-AUC (+ by gender), reference predictions
  utils.py          # seeding, device, log file
  visualize.py      # pitch plots, channel plots, data overview
tests/              # renderer vs. reference, mirror = row flip, models, sampler
results/plots/      # figures shown above
data/, models/      # downloaded data and trained weights (not in the repo)
```

## Acknowledgements

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Pitch plots:
[mplsoccer](https://github.com/andrewRowlinson/mplsoccer).
