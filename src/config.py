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
