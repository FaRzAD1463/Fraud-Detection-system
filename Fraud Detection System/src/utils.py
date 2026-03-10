import json
from src.config import THRESHOLD_PATH

def load_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        return json.load(f)["threshold"]