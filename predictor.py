import joblib
import pandas as pd
from config import MODEL_PATH, FEATURES

class CropPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run `python train.py` first."
            )
        self.pipeline = joblib.load(self.model_path)
        return self

    def predict(self, values, top_n=3):
        if self.pipeline is None:
            self.load()
        frame = pd.DataFrame([{key: values[key] for key in FEATURES}], columns=FEATURES)
        probabilities = self.pipeline.predict_proba(frame)[0]
        classes = self.pipeline.classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        return {
            "recommended_crop": str(ranked[0][0]),
            "confidence": round(float(ranked[0][1]), 4),
            "top_recommendations": [
                {"crop": str(label), "probability": round(float(prob), 4)}
                for label, prob in ranked[:top_n]
            ],
            "disclaimer": "Demo model output; validate with local agronomic expertise."
        }
