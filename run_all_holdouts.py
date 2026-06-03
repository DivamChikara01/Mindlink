import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train_mindlink_model import process_subject

files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))

rows = []
for file in files:
    rows.extend(process_subject(file))

df = pd.DataFrame(rows).dropna()

subjects = sorted(df["subject"].unique())
results = []

for holdout_subject in subjects:
    train_df = df[df["subject"] != holdout_subject]
    test_df = df[df["subject"] == holdout_subject]

    X_train = train_df.drop(columns=["label", "subject", "start_sec"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "subject", "start_sec"])
    y_test = test_df["label"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        target_names=["baseline", "stress"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, pred)

    results.append({
        "subject": holdout_subject,
        "test_windows": len(test_df),
        "accuracy": accuracy_score(y_test, pred),
        "baseline_precision": report["baseline"]["precision"],
        "baseline_recall": report["baseline"]["recall"],
        "baseline_f1": report["baseline"]["f1-score"],
        "stress_precision": report["stress"]["precision"],
        "stress_recall": report["stress"]["recall"],
        "stress_f1": report["stress"]["f1-score"],
        "false_stress_alerts": int(cm[0][1]),
        "missed_stress_windows": int(cm[1][0]),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("stress_f1", ascending=False)

print(results_df)

results_df.to_csv("results/all_subject_holdout_results.csv", index=False)
print("\nSaved: results/all_subject_holdout_results.csv")