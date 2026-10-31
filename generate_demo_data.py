"""Generate a reproducible synthetic dataset for local software demonstration only."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
CROPS = {
    "rice":   ([70, 35, 40, 24, 85, 6.2, 220], [18, 12, 14, 3, 8, .5, 55]),
    "maize":  ([90, 45, 35, 23, 65, 6.3, 85], [20, 14, 12, 4, 12, .6, 35]),
    "chickpea":([40, 65, 80, 20, 45, 7.0, 65], [15, 16, 18, 4, 12, .6, 30]),
    "kidneybeans":([20, 65, 20, 20, 65, 5.8, 105], [10, 15, 10, 3, 12, .5, 35]),
    "pigeonpeas":([20, 65, 20, 28, 50, 6.5, 100], [10, 15, 10, 4, 13, .6, 35]),
    "mothbeans":([20, 45, 20, 29, 55, 6.8, 35], [10, 14, 10, 4, 14, .6, 18]),
    "mungbean":([20, 45, 20, 28, 80, 6.8, 50], [10, 14, 10, 3, 10, .5, 20]),
    "blackgram":([40, 60, 20, 30, 65, 7.0, 65], [14, 15, 10, 3, 12, .5, 25]),
    "lentil":([20, 65, 20, 24, 65, 6.8, 45], [10, 15, 10, 4, 12, .5, 20]),
    "pomegranate":([20, 10, 40, 22, 90, 6.5, 110], [10, 8, 15, 5, 10, .6, 35]),
    "banana":([100, 80, 50, 27, 80, 6.2, 105], [20, 18, 18, 3, 10, .5, 30]),
    "mango":([20, 25, 30, 30, 50, 5.8, 95], [10, 12, 14, 4, 14, .7, 35]),
    "grapes":([20, 125, 200, 24, 80, 6.0, 70], [10, 22, 25, 4, 12, .6, 30]),
    "watermelon":([100, 10, 50, 26, 85, 6.5, 55], [18, 8, 18, 4, 10, .5, 25]),
    "muskmelon":([100, 15, 50, 27, 90, 6.5, 35], [18, 10, 18, 4, 9, .5, 18]),
    "apple":([20, 130, 200, 22, 90, 6.0, 115], [10, 22, 25, 5, 10, .6, 35]),
    "orange":([20, 20, 10, 24, 90, 7.0, 110], [10, 12, 8, 4, 10, .6, 35]),
    "papaya":([50, 60, 50, 33, 90, 6.5, 145], [15, 18, 18, 4, 8, .5, 40]),
    "coconut":([20, 10, 30, 27, 95, 5.8, 170], [10, 8, 14, 3, 5, .5, 40]),
    "cotton":([120, 45, 20, 24, 80, 6.5, 80], [22, 15, 10, 4, 12, .6, 35]),
    "jute":([80, 45, 40, 25, 85, 6.5, 175], [18, 15, 15, 3, 8, .5, 45]),
    "coffee":([100, 20, 30, 24, 60, 6.0, 155], [20, 12, 14, 4, 15, .6, 40]),
}

def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for crop, (means, stds) in CROPS.items():
        samples = rng.normal(loc=means, scale=stds, size=(100, len(means)))
        samples[:, :3] = np.clip(samples[:, :3], 0, 300)
        samples[:, 3] = np.clip(samples[:, 3], -5, 55)
        samples[:, 4] = np.clip(samples[:, 4], 0, 100)
        samples[:, 5] = np.clip(samples[:, 5], 3, 10)
        samples[:, 6] = np.clip(samples[:, 6], 0, 500)
        for row in samples:
            rows.append([*row.round(2), crop])
    columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    out = Path(__file__).resolve().parents[1] / "data" / "crop_recommendation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
    print(f"Wrote {len(rows)} synthetic records to {out}")

if __name__ == "__main__":
    main()
