import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load your main dataset
df = pd.read_csv("data/raw/trips.csv")

# OPTIONAL: adjust if your target column is different
target = "churned"

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target] if target in df.columns else None
)

os.makedirs("data/processed", exist_ok=True)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Train and test files created successfully!")