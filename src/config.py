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
    # lr: head / new layers; backbone_lr: pretrained layers (None -> same as lr)
    "small_cnn": {"lr": 1e-3, "backbone_lr": None},
    "resnet18": {"lr": 1e-3, "backbone_lr": 1e-4, "maxpool": True},
    "resnet18_nomaxpool": {"lr": 1e-3, "backbone_lr": 1e-4, "maxpool": False},
}
