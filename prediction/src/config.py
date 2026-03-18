"""Configuration for prediction service."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Model checkpoint paths
DAM_V2_CHECKPOINTS = MODELS_DIR / "dam_v2" / "checkpoints"
RTM_CHECKPOINTS = MODELS_DIR / "rtm" / "checkpoints"
SPIKE_CHECKPOINTS = MODELS_DIR / "spike" / "checkpoints"
DELTA_SPREAD_MODELS = MODELS_DIR / "delta-spread" / "models"

# InfluxDB configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "0691bd05e35a51b2")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ercot")

# Settlement points
SETTLEMENT_POINTS = [
    "HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST",
    "LZ_HOUSTON", "LZ_NORTH", "LZ_SOUTH", "LZ_WEST"
]

# Default prediction horizons
DAM_HORIZONS = list(range(1, 25))  # 1-24 hours
RTM_SHORT_HORIZONS = [1, 2, 3, 4]  # 15, 30, 45, 60 minutes
RTM_MED_HORIZONS = [1, 2, 4, 6, 12, 24]  # hours
