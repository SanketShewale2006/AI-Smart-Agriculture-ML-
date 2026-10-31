from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_recommender.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
