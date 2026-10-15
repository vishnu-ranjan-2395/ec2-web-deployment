
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(iris.data, iris.target)

joblib.dump(model, MODEL_PATH)

print(f"Model trained and saved to {MODEL_PATH}")
