
# MindLink SVM Upgrade

Your model comparison showed SVM RBF is better than Random Forest:

- SVM RBF mean F1: about 0.833
- SVM RBF stress recall: about 0.865
- Random Forest mean F1: about 0.786
- Random Forest stress recall: about 0.752

## How to use

Put these files inside your existing `mindlink_ai_starter` folder:

- `train_svm_model.py`
- `run_all_holdouts_svm.py`
- `make_svm_dashboard.py`

## Step 1: Train SVM model

```bash
python train_svm_model.py
```

This creates:

```text
models/mindlink_svm_rbf.joblib
models/mindlink_feature_columns.joblib
```

## Step 2: Create SVM dashboard file

```bash
python make_svm_dashboard.py
```

This creates:

```text
mindlink_dashboard_v2_svm.py
```

## Step 3: Run dashboard

```bash
streamlit run mindlink_dashboard_v2_svm.py
```

## Step 4: Run SVM all-subject holdouts

```bash
python run_all_holdouts_svm.py
```

This saves:

```text
results/all_subject_holdout_results_svm.csv
```

## Honest research claim

Good claim:

> I compared several models with subject-independent validation. SVM RBF performed best, especially for stress recall, so I upgraded the MindLink model from Random Forest to SVM RBF.

Do not claim this is medical-grade or exact emotion detection.
