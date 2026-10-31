import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, MODEL_PATH, FEATURES, TARGET

def train(data_path=DATA_PATH):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run `python scripts/generate_demo_data.py`."
        )
    df = pd.read_csv(data_path)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[required].dropna(subset=[TARGET])
    if df[TARGET].nunique() < 2:
        raise ValueError("At least two crop classes are required.")

    X, y = df[FEATURES], df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocess = ColumnTransformer([("numeric", numeric, FEATURES)])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "dataset_rows": int(len(df)),
        "features": FEATURES,
        "warning": "Synthetic demo dataset scores are not evidence of real-world performance."
    }
    metrics_path = MODEL_PATH.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to training CSV")
    args = parser.parse_args()
    train(args.data)
