# AI Smart Agriculture — Crop Recommendation ML

A complete starter project that recommends a suitable crop from soil nutrients and local climate measurements.

## Features
- Trainable scikit-learn classification pipeline
- Synthetic demo dataset (2,200 rows) so the project runs immediately
- Replaceable CSV dataset support
- Flask web interface and JSON prediction API
- Input validation and friendly error messages
- Saved model artifact, training metrics, and confusion matrix
- Unit tests and reproducible setup

> **Important:** The included dataset is synthetic and is for demonstrating the software pipeline only. It is not agronomic advice and must not be used to make real planting decisions. For real deployment, train and validate with trusted local soil, climate, and crop-yield data, and consult local agricultural experts.

## Inputs
- N, P, K: soil macronutrient values (mg/kg in the dataset convention)
- temperature: °C
- humidity: %
- ph: soil pH
- rainfall: mm (dataset convention; use a consistent measurement window)

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_demo_data.py
python train.py
python app.py
```

Open http://127.0.0.1:5000.

## Use your own dataset

Create `data/crop_recommendation.csv` with these exact columns:

`N,P,K,temperature,humidity,ph,rainfall,label`

Each row should contain measured features and the crop label. Keep units and measurement periods consistent. Then run:

```bash
python train.py --data data/crop_recommendation.csv
```

The training script performs a stratified train/test split, fits preprocessing and a Random Forest classifier, prints a classification report, and saves the pipeline to `models/crop_recommender.joblib`.

## API

`POST /api/predict`

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 24.5,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 200
}
```

Example response:

```json
{
  "recommended_crop": "rice",
  "confidence": 0.91,
  "top_recommendations": [
    {"crop": "rice", "probability": 0.91},
    {"crop": "jute", "probability": 0.04}
  ],
  "disclaimer": "Demo model output; validate with local agronomic expertise."
}
```

Confidence values are model scores, not guarantees of crop success.

## Project layout

```text
AI-Smart-Agriculture-ML/
├── app.py
├── train.py
├── requirements.txt
├── config.py
├── data/
│   └── crop_recommendation.csv
├── models/
├── scripts/
│   └── generate_demo_data.py
├── src/
│   ├── __init__.py
│   ├── predictor.py
│   └── validation.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── tests/
    └── test_validation.py
```

## Model and responsible use
- Baseline: median imputation, feature scaling, Random Forest classifier.
- Evaluation: held-out stratified test split; accuracy and per-class precision/recall/F1.
- Do not interpret synthetic-data scores as real-world performance.
- Before field use, use representative regional data, temporal/geographic holdouts, calibration, agronomist review, and monitoring for data drift.
- Never expose the Flask development server directly to the public internet. Use a production WSGI server, HTTPS, authentication, rate limits, and managed secrets for deployment.

## License
MIT — see `LICENSE`.
