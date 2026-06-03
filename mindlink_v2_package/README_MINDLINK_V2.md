# MindLink v2 Package

This package upgrades your current MindLink prototype.

## What this adds

1. `mindlink_dashboard_v2.py`
   - Personalized baseline deviation score
   - Stress score over time
   - Main sensor reason for alert
   - Color signal: GREEN / YELLOW / ORANGE / RED
   - Caregiver message
   - Top deviating features

2. `verify_mindlink_model.py`
   - Real subject-independent validation
   - Dumb baseline test
   - Shuffled-label test
   - Feature importance
   - Saves report files to `results/`

3. `holdout_subject_test.py`
   - Trains on all subjects except one
   - Tests on the held-out subject
   - More honest for showing whether it generalizes

4. `compare_models.py`
   - Compares Logistic Regression, SVM, Random Forest, and Gradient Boosting

## Installation

Put these files inside your existing `mindlink_ai_starter` folder.

Install Streamlit:

```bash
pip install streamlit
```

Your folder should look like:

```text
mindlink_ai_starter/
  train_mindlink_model.py
  mindlink_dashboard_v2.py
  verify_mindlink_model.py
  holdout_subject_test.py
  compare_models.py
  data/
    WESAD/
  models/
    mindlink_random_forest.joblib
    mindlink_feature_columns.joblib
```

## Run the v2 dashboard

```bash
streamlit run mindlink_dashboard_v2.py
```

## Run model verification

```bash
python verify_mindlink_model.py
```

This creates:

```text
results/validation_scores.csv
results/confusion_matrix.csv
results/classification_report.csv
results/feature_importance.csv
results/validation_summary.md
```

## Run fair holdout test

```bash
python holdout_subject_test.py S2
```

You can replace `S2` with any subject.

## Run model comparison

```bash
python compare_models.py
```

## What to say honestly

MindLink v2 is a personalized baseline prototype. It does not detect exact emotions. It detects physiological baseline deviation and stress-pattern similarity.

Strong description:

> MindLink learns an individual's baseline from wearable physiological signals and translates deviations into simple caregiver-facing color signals.

Weak/wrong description:

> MindLink tells exactly how an autistic child feels.

Do not use the weak version.
