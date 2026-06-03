from joblib import load
import pandas as pd
from train_mindlink_model import process_subject

model = load("models/mindlink_random_forest.joblib")
feature_columns = load("models/mindlink_feature_columns.joblib")

# Change S2 to any subject folder you want
rows = process_subject("data/WESAD/S2/S2.pkl")

df = pd.DataFrame(rows).dropna()

# Pick one window
sample = df.iloc[0]

X = pd.DataFrame([sample[feature_columns]], columns=feature_columns)

prediction = int(model.predict(X)[0])
stress_score = float(model.predict_proba(X)[0][1]) * 100

print("Subject:", sample["subject"])
print("Start second:", sample["start_sec"])
print("True label:", "stress" if sample["label"] == 1 else "baseline")
print("Prediction:", "stress-pattern change" if prediction == 1 else "baseline")
print("Stress score:", round(stress_score, 2), "%")