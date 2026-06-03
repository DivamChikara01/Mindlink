
"""
Run all-subject holdout testing using SVM RBF.
Run inside mindlink_ai_starter:
    python run_all_holdouts_svm.py
"""
from __future__ import annotations

import glob
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from train_mindlink_model import process_subject


def main() -> None:
    files = sorted(glob.glob('data/WESAD/S*/S*.pkl'))
    if not files:
        raise FileNotFoundError('No WESAD files found. Expected: data/WESAD/S2/S2.pkl')

    rows = []
    for file in files:
        print('Loading', file)
        rows.extend(process_subject(file))

    df = pd.DataFrame(rows).dropna()
    subjects = sorted(df['subject'].unique())
    results = []

    for holdout_subject in subjects:
        train_df = df[df['subject'] != holdout_subject]
        test_df = df[df['subject'] == holdout_subject]

        X_train = train_df.drop(columns=['label', 'subject', 'start_sec'])
        y_train = train_df['label']
        X_test = test_df.drop(columns=['label', 'subject', 'start_sec'])
        y_test = test_df['label']

        model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        report = classification_report(
            y_test, pred, target_names=['baseline', 'stress'], output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, pred)

        results.append({
            'subject': holdout_subject,
            'test_windows': len(test_df),
            'accuracy': accuracy_score(y_test, pred),
            'baseline_precision': report['baseline']['precision'],
            'baseline_recall': report['baseline']['recall'],
            'baseline_f1': report['baseline']['f1-score'],
            'stress_precision': report['stress']['precision'],
            'stress_recall': report['stress']['recall'],
            'stress_f1': report['stress']['f1-score'],
            'false_stress_alerts': int(cm[0][1]),
            'missed_stress_windows': int(cm[1][0]),
        })

    results_df = pd.DataFrame(results).sort_values('stress_f1', ascending=False)
    print('\n===== ALL SUBJECT HOLDOUT RESULTS: SVM RBF =====')
    print(results_df)
    Path('results').mkdir(exist_ok=True)
    results_df.to_csv('results/all_subject_holdout_results_svm.csv', index=False)
    print('\nSaved: results/all_subject_holdout_results_svm.csv')

if __name__ == '__main__':
    main()
