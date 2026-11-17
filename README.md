# xG-Vision: Expected Goals from bird's-eye freeze frames

Classic expected-goals (xG) models use hand-crafted features such as distance and angle to goal.
**xG-Vision** asks whether a convolutional network can learn shot quality directly from a
picture of the situation: every shot is turned into a multi-channel bird's-eye "image" of the
attacking half (the shooter, teammates, defenders, goalkeeper and the shooting cone), built from
the player positions StatsBomb records at the moment of the shot (*freeze frames*).

> **Status:** steps 1–3 are done: data pipeline, CNN training, and a test-set evaluation
> against StatsBomb xG and logistic-regression baselines (match-level bootstrap CIs,
> calibration, Grad-CAM). **Headline:** from the pitch picture plus the body part, a 1.2 M
> parameter CNN gets within 0.005 log loss of StatsBomb's xG and clearly beats distance/angle
> baselines, but StatsBomb's model is still significantly better on the test split.

## Roadmap

| Step | Content | Status |
|---|---|---|
| 1 | Data pipeline: StatsBomb Open Data → cleaned shot table, match-level split, multi-channel pitch tensors, PyTorch `Dataset`, sanity plots | ✅ done |
| 2 | Models: small CNN from scratch vs. ImageNet ResNet18 (6-channel first conv), body part as an extra input, training loop with early stopping | ✅ done |
| 3 | Evaluation on the test split: log loss / Brier / ROC-AUC, calibration, comparison with StatsBomb xG and a distance + angle logistic regression, interesting shots, Grad-CAM | ✅ done |
| 4 | Sanity tests: move the shooter, add a defender in the shooting lane, remove the goalkeeper; xG heat map over the box | ⏳ next |
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

## Results (test split)

All choices (models, epochs, Grad-CAM layer) were fixed on the validation split first; the
test split was evaluated once. Test: 14,166 open-play shots from 597 matches, goal rate 0.1105.
95% intervals come from 2,000 bootstrap resamples **of matches** (shots of one match are not
independent). CNN rows are the seed-42 runs.

| Model | Log loss | Brier | ROC-AUC | Log loss, 3 seeds (mean ± std) |
|---|---|---|---|---|
| StatsBomb xG | **0.2778** [0.267, 0.289] | **0.0799** | **0.807** [0.795, 0.818] | – |
| SmallCNN | 0.2826 [0.272, 0.293] | 0.0813 | 0.797 [0.784, 0.809] | 0.2824 ± 0.0006 |
| SmallCNN, no distance map | 0.2830 | 0.0818 | 0.797 | 0.2828 ± 0.0009 |
| ResNet18, random init | 0.2832 | 0.0814 | 0.797 | (1 seed) |
| ResNet18, ImageNet | 0.2872 | 0.0826 | 0.796 | (1 seed) |
| LogReg: distance + angle + body part | 0.3044 | 0.0879 | 0.753 | – |
| LogReg: distance + angle | 0.3131 | 0.0903 | 0.728 | – |
| Constant (training goal rate) | 0.3478 | 0.0983 | 0.500 | – |

Paired match-bootstrap differences (a − b, 95% CI):

| Comparison | Δ log loss | Δ ROC-AUC | Reading |
|---|---|---|---|
| SmallCNN − StatsBomb xG | +0.0047 [+0.0023, +0.0071] | −0.010 [−0.015, −0.005] | StatsBomb is better (also within men's and women's matches) |
| SmallCNN − LogReg (dist + angle + body) | −0.0218 [−0.0258, −0.0179] | +0.043 [+0.035, +0.051] | the CNN learns much more than distance and angle |
| SmallCNN − ResNet18 (ImageNet) | −0.0046 [−0.0075, −0.0017] | +0.000 [−0.005, +0.006] | the small CNN is better calibrated; same ranking |
| ResNet18 ImageNet − random init | +0.0039 [+0.0013, +0.0066] | −0.000 [−0.005, +0.004] | ImageNet pretraining does **not** help (see below) |
| SmallCNN − no distance map | −0.0005 [−0.0023, +0.0015] | −0.001 [−0.004, +0.003] | no meaningful difference |

By gender (SmallCNN / StatsBomb, log loss): men 0.2752 / 0.2704, women 0.2960 / 0.2914.
Full tables, including Brier, per-gender metrics and the bootstrap details:
[`results/metrics_test.md`](results/metrics_test.md) (validation: [`results/metrics_val.md`](results/metrics_val.md)).

**What this means**

- **The picture carries most of the signal.** With only player positions and the body part, the
  CNN beats a distance + angle + body-part logistic regression by 0.022 log loss and 0.043 AUC.
- **StatsBomb xG remains ahead** (0.005 log loss, 0.010 AUC, both significant). Its model also
  uses inputs the CNN does not get, such as shot technique (volley, lob…) and the first-time,
  one-on-one and under-pressure flags. Among the 5% of shots where the CNN is much *higher* than
  StatsBomb, non-"Normal" techniques (volleys, half-volleys, lobs) are more common than overall
  (37% vs. 23%): information the picture does not contain.
- **Pretraining did not help.** ImageNet ResNet18 is worse than the same network from random
  initialisation in log loss and Brier (AUC identical). The difference exceeds 0.003 but comes
  from a single seed and a different learning-rate setup (1e-4 on the pretrained layers), so the
  safe reading is "ImageNet features bring no benefit for these synthetic images". The 1.2 M
  parameter SmallCNN is as good as either ResNet18.
- **The distance map channel is redundant**: without it the SmallCNN is equally good (3 seeds
  each). The network infers position from the shooter blob and the shot cone.
- **Seeds hardly matter**: three SmallCNN seeds differ by ±0.0006 log loss on test.
- **Test-time mirroring** (average of the shot and its mirror image) gives a small gain
  (SmallCNN 0.2826 → 0.2822); it is reported separately and not used above.
- The val → test drop (StatsBomb 0.263 → 0.278) affects all models alike: the test split has a
  higher goal rate (11.1% vs. 10.3%), which raises log loss for every model.

### Calibration

Equal-count bins on log axes; error bars are match-bootstrap 95% intervals. All models are well
calibrated (ECE: SmallCNN 0.009, ResNet18 0.011, StatsBomb 0.012, LogReg 0.015). The CNN's
predictions are spread more widely than the logistic regression's, which is where its better
ranking comes from. ImageNet ResNet18 puts ~6% of the shots below 0.005 xG and is slightly
overconfident there. Per-gender panels:
[`calibration_test_by_gender.png`](results/plots/calibration_test_by_gender.png).

![Calibration on the test split](results/plots/calibration_test.png)

### Where the CNN and StatsBomb disagree (with Grad-CAM)

For each direction, the 3 goals and 3 non-goals with the largest logit difference between the
SmallCNN and StatsBomb xG. Overlay: Grad-CAM of SmallCNN block 2 (20 × 15 feature map, chosen on
validation shots, see [`gradcam_layers_val.png`](results/plots/gradcam_layers_val.png)).

![CNN much higher than StatsBomb](results/plots/interesting_cnn_higher_test.png)
![CNN much lower than StatsBomb](results/plots/interesting_cnn_lower_test.png)

Typical cases: StatsBomb rates some close-range half-volleys and first-time shots very low,
while the CNN, which only sees the positions, rates them high. Conversely, shots that StatsBomb
rates near 1.0 (an empty net, one-on-ones) are capped around 0.7–0.8 by the CNN.
Details: [`results/interesting_test.md`](results/interesting_test.md).

**Grad-CAM caveat.** The maps are of limited use here. Because the network ends in global
average pooling, it also draws evidence from *empty* areas, and the maps often highlight the
free space around the player cluster rather than single players. Grad-CAM also only shows
positive evidence (for a goal), so a defender who lowers xG never lights up, and when nothing
pushes the logit up, the map is empty. Step 4 tests the model's behaviour directly by moving
players.

### Step 2: training (validation split)

| Model | Val log loss | Best / last epoch | Training time |
|---|---|---|---|
| SmallCNN (3 seeds) | 0.2654 ± 0.0002 | 9–11 / 14–16 | 11–13 min |
| SmallCNN, no distance map (3 seeds) | 0.2655 ± 0.0005 | 8–11 / 13–16 | 11–13 min |
| ResNet18, random init | 0.2654 | 9 / 14 | 12 min |
| ResNet18, ImageNet | 0.2690 | 5 / 10 | 8 min |

The pretrained ResNet18 starts to overfit after epoch 5, while the small CNN improves steadily:

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
python -m src.train --model small_cnn --seed 43                  # other seeds -> models/small_cnn_s43/
python -m src.train --model resnet18_scratch                     # ablations (also: small_cnn_nodist)

python -m src.baselines                        # logistic-regression baselines on val
python -m src.evaluate --split val             # full report: metrics, bootstrap CIs, calibration plots
python -m src.interpret --compare-layers       # Grad-CAM at several layers (val), to choose GRADCAM_LAYER
python -m src.evaluate --split test            # same report on the test split (run once, at the end)
python -m src.interpret --split test           # disagreement shots with Grad-CAM

python -m pytest tests -q                      # tests needing data/shots.pkl are skipped without it
```

Each run writes `models/<run>/` (`best.pt`, `last.pt`, `history.csv`, `train.log`,
`val_predictions.csv`) and `results/plots/training_curves_<run>.png`. The evaluation writes
`results/metrics_<split>.md|csv`, `results/interesting_<split>.md` and the plots in `results/plots/`.

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
  baselines.py      # logistic regressions on distance + angle (+ body part)
  evaluate.py       # metrics (+ by gender), match-level bootstrap, calibration, reports
  gradcam.py        # Grad-CAM for the SmallCNN
  interpret.py      # Grad-CAM layer comparison, shots where CNN and StatsBomb disagree
  utils.py          # seeding, device, log file
  visualize.py      # pitch plots, channel plots, data overview
tests/              # renderer vs. reference, mirror = row flip, models, sampler, bootstrap/AUC, Grad-CAM
results/            # metric reports (markdown + csv); plots/ holds the figures shown above
data/, models/      # downloaded data and trained weights (not in the repo)
```

## Acknowledgements

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Pitch plots:
[mplsoccer](https://github.com/andrewRowlinson/mplsoccer).
