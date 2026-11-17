"""Central configuration for the CNN-xG project."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (on Colab, point PROJECT_ROOT to a Google Drive folder so the cache
# survives runtime resets, e.g. Path("/content/drive/MyDrive/xg-cnn"))
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"            # per-match extracted shots (json)
SHOTS_PATH = DATA_DIR / "shots.pkl"       # cleaned shot table (one row per shot)
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

STATSBOMB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
GENDERS = ("male", "female")       # keep both; gender is stored so we can filter later
DOWNLOAD_WORKERS = 16

# Shot types used for training. Penalties / direct free kicks / direct corners
# are kept in shots.pkl but filtered out here.
TRAIN_SHOT_TYPES = ("Open Play",)

# ---------------------------------------------------------------------------
# Pitch geometry (StatsBomb coordinates, yards). Every event is normalised so
# the shooting team attacks left -> right, i.e. towards x = 120.
# ---------------------------------------------------------------------------
PITCH_LENGTH, PITCH_WIDTH = 120.0, 80.0
GOAL_X = 120.0
POST_LOW_Y, POST_HIGH_Y = 36.0, 44.0          # goal is 8 yards wide
GOAL_CENTER = (GOAL_X, 40.0)

# Region rendered into the tensor: the attacking half.
X_MIN, X_MAX = 60.0, 120.0
Y_MIN, Y_MAX = 0.0, 80.0

# Resolution: pixels per yard. 2 -> 0.5 yd/px -> tensor of 160 (H, y-axis) x 120 (W, x-axis)
PX_PER_YARD = 2
IMG_H = int((Y_MAX - Y_MIN) * PX_PER_YARD)    # 160
IMG_W = int((X_MAX - X_MIN) * PX_PER_YARD)    # 120

# Gaussian blob radius for each player (yards)
PLAYER_SIGMA = 1.0

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
CHANNELS = (
    "shooter",        # 0: the shot taker
    "teammates",      # 1: attacking teammates (no shooter)
    "opponents",      # 2: defending outfield players
    "goalkeeper",     # 3: defending goalkeeper
    "shot_cone",      # 4: triangle shooter -> both posts (binary)
    "goal_distance",  # 5: static map, distance of every pixel to goal centre (CoordConv-style)
)
N_CHANNELS = len(CHANNELS)

# ---------------------------------------------------------------------------
# Match-level split
# ---------------------------------------------------------------------------
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

# ---------------------------------------------------------------------------
# Scalar ("extra") model inputs, concatenated to the CNN features before the head.
# Body part is one-hot over these groups; left/right foot are merged so that
# the y-mirror augmentation stays valid.
# ---------------------------------------------------------------------------
BODY_PART_GROUPS = ("head", "foot", "other")
BODY_PART_MAP = {"Head": "head", "Right Foot": "foot", "Left Foot": "foot", "Other": "other"}
N_EXTRA_FEATURES = len(BODY_PART_GROUPS)

# ---------------------------------------------------------------------------
# Training (step 2). Per-model overrides live in MODEL_CONFIGS.
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"          # checkpoints + history (not committed)

BATCH_SIZE = 256
NUM_WORKERS = 6
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 5                       # epochs without val log-loss improvement
WEIGHT_DECAY = 1e-4
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 2
AMP_DTYPE = "bfloat16"                        # "bfloat16", "float16" or "none"
BN_RECAL_BATCHES = 40                         # "precise BN" before every validation (see train.py)
HEAD_HIDDEN = 64
HEAD_DROPOUT = 0.3

MODEL_CONFIGS = {
    # arch: network; lr: head / new layers; backbone_lr: pretrained layers (None -> same as lr);
    # channels: input channels the model sees (None -> all, see CHANNELS); pretrained: ImageNet weights
    "small_cnn": {"arch": "small_cnn", "lr": 1e-3, "backbone_lr": None},
    "small_cnn_nodist": {"arch": "small_cnn", "lr": 1e-3, "backbone_lr": None,
                         "channels": (0, 1, 2, 3, 4)},                 # ablation: no goal-distance map
    "resnet18": {"arch": "resnet18", "lr": 1e-3, "backbone_lr": 1e-4, "maxpool": True, "pretrained": True},
    "resnet18_scratch": {"arch": "resnet18", "lr": 1e-3, "backbone_lr": None, "maxpool": True,
                         "pretrained": False},                        # same net, random init
    "resnet18_nomaxpool": {"arch": "resnet18", "lr": 1e-3, "backbone_lr": 1e-4, "maxpool": False,
                           "pretrained": True},
}

# ---------------------------------------------------------------------------
# Evaluation (step 3). Fixed on the validation split BEFORE the test split is used.
# Prediction columns are named by run directory (models/<run>/best.pt) or baseline key.
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"

EVAL_MODELS = (                               # trained CNN runs in the main tables (seed 42)
    "small_cnn", "resnet18", "resnet18_scratch", "small_cnn_nodist",
)
SEED_GROUPS = {                               # extra seeds -> mean +- std columns
    "small_cnn": ("small_cnn", "small_cnn_s43", "small_cnn_s44"),
    "small_cnn_nodist": ("small_cnn_nodist", "small_cnn_nodist_s43", "small_cnn_nodist_s44"),
}
TTA_MODELS = ("small_cnn", "resnet18")        # extra rows: mean of original + mirrored prediction
BASELINES = ("statsbomb_xg", "constant", "logreg_dist_angle", "logreg_dist_angle_body")

DISPLAY_NAMES = {
    "small_cnn": "SmallCNN",
    "resnet18": "ResNet18 (ImageNet)",
    "resnet18_scratch": "ResNet18 (random init)",
    "small_cnn_nodist": "SmallCNN, no distance map",
    "statsbomb_xg": "StatsBomb xG",
    "constant": "Constant (train goal rate)",
    "logreg_dist_angle": "LogReg: distance + angle",
    "logreg_dist_angle_body": "LogReg: distance + angle + body part",
}

BOOTSTRAP_B = 2000                            # match-level bootstrap resamples
BOOTSTRAP_SEED = 2026
BOOTSTRAP_PAIRS = (                           # paired differences (a - b), seed-42 models
    ("small_cnn", "statsbomb_xg"),
    ("small_cnn", "resnet18"),
    ("small_cnn", "logreg_dist_angle_body"),
    ("resnet18", "resnet18_scratch"),
    ("small_cnn", "small_cnn_nodist"),
)
CALIBRATION_MODELS = ("small_cnn", "resnet18", "statsbomb_xg", "logreg_dist_angle_body")
CALIBRATION_BINS = 15                         # equal-count bins

INTERESTING_MODEL = "small_cnn"               # compared with StatsBomb xG for the example shots
INTERESTING_PER_GROUP = 3                     # per direction: 3 goals + 3 non-goals
GRADCAM_LAYER = 2      # SmallCNN block (20x15 map), chosen on val with `python -m src.interpret --compare-layers`

# ---------------------------------------------------------------------------
# Sanity tests and heat maps (step 4). SmallCNN seed 42 on the test split.
# ---------------------------------------------------------------------------
SANITY_MODEL = "small_cnn"
SANITY_STEP = 3.0                             # yd: shooter closer/farther, goalkeeper sideways/forward
SANITY_DEF_NEAR = 1.5                         # yd in front of the shooter (test 2a)
SANITY_ASIDE = 6.0                            # yd beside the shot line for the control defender (2c)
SANITY_UNCHANGED = 1e-3                       # |delta p| below this counts as "no change"
DISTANCE_BANDS = (0, 8, 12, 18, 25, 200)      # yd, for the per-band breakdown
SANITY_EXAMPLE_SEED = 7                       # rule-based choice of the example shots
HEATMAP_X = (78.0, 119.0)                     # shooter grid (inclusive), 1 yd steps; x <= 119 keeps 1 yd off the goal line
HEATMAP_Y = (10.0, 70.0)
HEATMAP_STEP = 1.0
