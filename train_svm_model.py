
"""
Train the upgraded MindLink SVM RBF model.
Run inside mindlink_ai_starter:
    python train_svm_model.py
"""
from __future__ import annotations

import glob
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
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
        print('Processing', file)
        rows.extend(process_subject(file))

    df = pd.DataFrame(rows).dropna()
    X = df.drop(columns=['label', 'subject', 'start_sec'])
    y = df['label']
    groups = df['subject']

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    )

    cv = GroupKFold(n_splits=min(5, df['subject'].nunique()))
    scores = cross_validate(
        model, X, y, groups=groups, cv=cv,
        scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
        n_jobs=-1,
    )

    print('\n===== SVM RBF SUBJECT-INDEPENDENT RESULTS =====')
    print('Mean F1:', round(float(scores['test_f1'].mean()), 4))
    print('Mean accuracy:', round(float(scores['test_accuracy'].mean()), 4))
    print('Mean stress recall:', round(float(scores['test_recall'].mean()), 4))

    pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)
    print('\nConfusion matrix:')
    print(confusion_matrix(y, pred))
    print('\nClassification report:')
    print(classification_report(y, pred, target_names=['baseline', 'stress'], zero_division=0))

    Path('models').mkdir(exist_ok=True)
    model.fit(X, y)
    dump(model, 'models/mindlink_svm_rbf.joblib')
    dump(list(X.columns), 'models/mindlink_feature_columns.joblib')
    print('\nSaved:')
    print('models/mindlink_svm_rbf.joblib')
    print('models/mindlink_feature_columns.joblib')

if __name__ == '__main__':
    main()
