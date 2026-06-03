# MindLink AI Starter Model

This is a starter AI prototype for MindLink.

## What it does

It trains a Random Forest model to classify wearable-sensor windows as:

- `0 = baseline`
- `1 = stress-pattern change`

This is NOT a medical diagnosis model. It is a prototype for detecting physiological stress-pattern changes.

## Dataset

Use WESAD. Put the dataset here:

```text
mindlink_ai_starter/
  data/
    WESAD/
      S2/
        S2.pkl
      S3/
        S3.pkl
      ...
```

## Install

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train_mindlink_model.py
```

After training, you should get:

```text
models/mindlink_random_forest.joblib
models/mindlink_feature_columns.joblib
```

## Test the saved model with fake sample input

```bash
python predict_demo.py
```

## What to say honestly

"I built a MindLink AI prototype with AI-assisted coding. It uses public wearable stress data and trains a Random Forest classifier to detect baseline vs stress-pattern changes. It is not a final medical device model yet."
# Mindlink
# Mindlink
