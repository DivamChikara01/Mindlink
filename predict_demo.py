"""
Demo prediction file for the saved MindLink model.

This uses fake example features only to prove loading/prediction works.
Real prediction requires creating features from actual wearable sensor windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


MODEL_PATH = Path("models/mindlink_random_forest.joblib")
COLUMNS_PATH = Path("models/mindlink_feature_columns.joblib")


def main() -> None:
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Train first with: python train_mindlink_model.py"
        )

    model = load(MODEL_PATH)
    feature_columns = load(COLUMNS_PATH)

    # Fake example: replace with real features from sensor data later.
    fake_features = {col: 0.0 for col in feature_columns}

    # Example values just so the code runs.
    for col in feature_columns:
        if "EDA" in col and "mean" in col:
            fake_features[col] = 0.5
        elif "TEMP" in col and "mean" in col:
            fake_features[col] = 32.0
        elif "BVP" in col and "std" in col:
            fake_features[col] = 10.0
        elif "ACC" in col and "std" in col:
            fake_features[col] = 0.1

    X_new = pd.DataFrame([fake_features], columns=feature_columns)

    pred = int(model.predict(X_new)[0])

    if hasattr(model, "predict_proba"):
        prob_stress = float(model.predict_proba(X_new)[0][1])
    else:
        prob_stress = np.nan

    label = "possible stress-pattern change" if pred == 1 else "baseline"

    print("Prediction:", label)
    print("Stress score:", round(prob_stress * 100, 2), "%")


if __name__ == "__main__":
    main()
