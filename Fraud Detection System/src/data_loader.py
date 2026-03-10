import pandas as pd
from src.config import DATA_PATH
from src.logger import get_logger

logger = get_logger()

def load_data():
    logger.info("Loading dataset")
    df = pd.read_csv(DATA_PATH)
    return df