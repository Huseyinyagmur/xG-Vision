---
license: other
license_name: statsbomb-public-data-user-agreement
license_link: https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf
library_name: pytorch
tags:
  - football
  - soccer
  - expected-goals
  - xg
  - sports-analytics
  - cnn
---

# xG-Vision SmallCNN: expected goals from a bird's-eye picture of the shot

![StatsBomb](statsbomb_logo.png)

A small convolutional network (1.2 M parameters) that estimates the probability that a shot
becomes a goal (**expected goals, xG**) from a 6-channel bird's-eye image of the attacking half
at the moment of the shot (shooter, teammates, defenders, goalkeeper, shooting cone, distance
map) plus the body part (foot / head / other). It is trained on the player positions StatsBomb
records for every shot ("freeze frames").

- Code, training and evaluation: https://github.com/Huseyinyagmur/xG-Vision
- Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). **Data provided by StatsBomb.**
- File: `small_cnn.pt` (PyTorch checkpoint: `state_dict`, `model_name`, `prior`, training config,
  validation metrics). Seed 42, epoch 9 of 14, selected on the validation split.

## How to use

```python
# git clone https://github.com/Huseyinyagmur/xG-Vision && cd xG-Vision && pip install -r requirements-app.txt
from huggingface_hub import hf_hub_download
from src.models import load_checkpoint
from src.perturb import Shot
import numpy as np, torch

model, _ = load_checkpoint(hf_hub_download("huseyinyagmur/xg-vision-smallcnn", "small_cnn.pt"))
# StatsBomb coordinates (120 x 80 yd, attacking towards x = 120); freeze frame rows: [x, y, teammate, is_goalkeeper]
ff = np.array([[119.0, 40.0, 0, 1], [110.0, 42.0, 0, 0]], np.float32)
shot = Shot(x=104.0, y=38.0, ff=ff, body="foot")
with torch.no_grad():
    xg = torch.sigmoid(model(torch.from_numpy(shot.image())[None], torch.from_numpy(shot.extra())[None]))
```

The interactive tactics board in the repository (`streamlit run app/streamlit_app.py`) downloads
this file automatically and averages each prediction with the one for the mirror image
(left/right flip), which is slightly better calibrated.

## Training data

- StatsBomb Open Data, all men's and women's competitions with event data (snapshot November
  2026): 3,961 matches, **95,160 open-play shots** with a freeze frame, goal rate 10.4%.
  Penalties, free kicks and shots from the own half are excluded.
- Split **by match** (stratified by competition): train 66,754 / validation 14,240 / test 14,166 shots.
- Mirror augmentation (y → 80 − y). Plain binary cross-entropy (no class weights, to keep the
  probabilities calibrated), AdamW, early stopping on the validation log loss, batch-norm
  statistics recomputed with frozen weights before every validation ("precise BN").

## Evaluation (test split, evaluated once after all choices were fixed on validation)

14,166 shots from 597 matches, goal rate 11.1%. 95% intervals: 2,000 bootstrap resamples of matches.

| Model | Log loss | Brier | ROC-AUC |
|---|---|---|---|
| StatsBomb xG (StatsBomb's own model) | **0.2778** [0.267, 0.289] | **0.0799** | **0.807** [0.795, 0.818] |
| **This model** | 0.2826 [0.272, 0.293] | 0.0813 [0.078, 0.085] | 0.797 [0.784, 0.809] |
| This architecture, 3 seeds | 0.2824 ± 0.0006 | 0.0813 | 0.798 ± 0.002 |
| Logistic regression: distance + angle + body part | 0.3044 | 0.0879 | 0.753 |
| Constant (training goal rate) | 0.3478 | 0.0983 | 0.500 |

- Against StatsBomb xG (paired, by match): log loss +0.0047 [+0.0023, +0.0071], ROC-AUC −0.010
  [−0.015, −0.005]. **StatsBomb's model is significantly better**; it also uses inputs this model
  does not see (shot technique, first-time, one-on-one, under pressure).
- Against the distance + angle + body part regression: log loss −0.022, ROC-AUC +0.043.
- Men / women (log loss): 0.2752 / 0.2960 (StatsBomb xG 0.2704 / 0.2914).
- **Calibration**: expected calibration error 0.009 (15 equal-count bins; StatsBomb xG 0.012).

![Calibration on the test split](calibration_test.png)

## Known limitations

Found with controlled edits of all 14,166 test shots (move the shooter, add defenders, move or
remove the goalkeeper); details in the repository README.

1. **Goal-line edge artefact.** Shots from within ~1.5 yd of the goal line outside the posts get
   far too high xG (≈ 0.45 even 28 yd wide; a logistic regression gives 0.02). Only ~110 training
   shots come from there. Do not use the model there.
2. **Goalkeeper position is confounded with goals.** Moving only the keeper off the line raises
   xG (mean 0.072 → 0.110 for a 4 yd step), because in the data a keeper off the line usually means
   a keeper out of position. The model has not learned the effect of narrowing the angle.
3. **Distance vs. pressure.** Stepping back 3 yd raises xG for 27% of shots, mostly when it takes
   the shooter away from the nearest defender; the size of the effect looks too large.
4. **A defender halfway along the shot line** sometimes raises xG (19% of shots), mostly 3–10 yd in
   front of the keeper.
5. **Left/right asymmetry** of up to 0.15 xG despite mirror augmentation (mirror averaging helps).
6. **Out of distribution:** no goalkeeper (0.1% of training shots), only a goalkeeper on the
   pitch, headers from 25+ yd, set pieces and penalties.
7. Only positions and body part are used: no shot technique, ball height, pass type or player skill.

## Intended use

Research, teaching and exploring how an image model reads shot situations (for example on the
interactive tactics board). **Not** for commercial use (the training data may not be used
commercially, see below), betting, or decisions about real players.

## Licence and attribution

- The code that produced this model is MIT-licensed (see the GitHub repository).
- The model was trained on **StatsBomb Open Data**, which is provided under the
  [StatsBomb Public Data User Agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
  That agreement does not allow commercial exploitation of the data or of analysis derived from
  it, and requires publications to credit StatsBomb with its logo. These weights are shared for
  non-commercial research and education under those terms. The raw data is not redistributed here.
