import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")