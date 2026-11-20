"""Configuration. API keys from environment or .env file."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# Player detection model (Roboflow hosted)
PLAYER_MODEL_ID = "soccer-player-ball-tracker-bechj/4"

# Detection
PLAYER_CONFIDENCE = 0.15

# Tracking
TRACKER_FRAME_RATE = 30
TRACKER_ACTIVATION_THRESHOLD = 0.25
