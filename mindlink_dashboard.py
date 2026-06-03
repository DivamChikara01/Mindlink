"""
MindLink Dashboard
Run with:
    streamlit run mindlink_dashboard.py

Prototype only:
- baseline vs stress-pattern change
- not emotion detection
- not medical diagnosis
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

from train_mindlink_model import process_subject


MODEL_PATH = Path("models/mindlink_random_forest.joblib")
COLUMNS_PATH = Path("models/mindlink_feature_columns.joblib")


def status_from_score(score: float) -> tuple[str, str]:
    if score >= 80:
        return (
            "Possible stress-pattern change",
            "Check in calmly. Reduce noise, pressure, and stimulation. Give the person space and a simple choice."
        )
    if score >= 60:
        return (
            "Rising physiological activation",
            "Monitor gently. Use a calm voice and look for environmental triggers."
        )
    if score >= 40:
        return (
            "Uncertain / mixed signal",
            "No strong alert. Keep observing and compare with behavior/context."
        )
    return (
        "Baseline-like pattern",
        "No stress-pattern alert from the model right now."
    )


def confidence_from_score(score: float) -> str:
    distance = abs(score - 50)
    if distance >= 35:
        return "High"
    if distance >= 20:
        return "Medium"
    return "Low"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        st.error("Model files not found. First run: python train_mindlink_model.py")
        st.stop()
    model = load(MODEL_PATH)
    feature_columns = load(COLUMNS_PATH)
    return model, feature_columns


@st.cache_data
def load_subject_rows(subject_path: str) -> pd.DataFrame:
    rows = process_subject(subject_path)
    return pd.DataFrame(rows).dropna()


def main() -> None:
    st.set_page_config(
        page_title="MindLink AI Dashboard",
        page_icon="🧠",
        layout="wide"
    )

    st.title("MindLink AI Prototype Dashboard")
    st.caption("Baseline vs stress-pattern change detection from wearable sensor windows.")

    model, feature_columns = load_model()
    subject_files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))

    if not subject_files:
        st.error("No WESAD subject files found. Expected path like: data/WESAD/S2/S2.pkl")
        st.stop()

    with st.sidebar:
        st.header("Controls")

        subject_path = st.selectbox("Choose WESAD subject", subject_files, index=0)

        df = load_subject_rows(subject_path)

        label_filter = st.selectbox(
            "Filter windows",
            ["All", "Baseline only", "Stress only"]
        )

        if label_filter == "Baseline only":
            shown_df = df[df["label"] == 0].copy()
        elif label_filter == "Stress only":
            shown_df = df[df["label"] == 1].copy()
        else:
            shown_df = df.copy()

        if shown_df.empty:
            st.warning("No windows found for this filter.")
            st.stop()

        window_index = st.slider(
            "Window number",
            min_value=0,
            max_value=len(shown_df) - 1,
            value=0
        )

        st.divider()
        st.write("Prototype only. Not medical advice. Not emotion detection.")

    sample = shown_df.iloc[window_index]
    X = pd.DataFrame([sample[feature_columns]], columns=feature_columns)

    prediction = int(model.predict(X)[0])
    stress_score = float(model.predict_proba(X)[0][1]) * 100

    status, caregiver_message = status_from_score(stress_score)
    confidence = confidence_from_score(stress_score)

    true_label = "Stress" if int(sample["label"]) == 1 else "Baseline"
    prediction_label = "Stress-pattern change" if prediction == 1 else "Baseline"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stress score", f"{stress_score:.2f}%")
    col2.metric("Prediction", prediction_label)
    col3.metric("True WESAD label", true_label)
    col4.metric("Confidence", confidence)

    st.progress(min(max(stress_score / 100, 0), 1))

    if stress_score >= 80:
        st.error(f"Status: {status}")
    elif stress_score >= 60:
        st.warning(f"Status: {status}")
    elif stress_score >= 40:
        st.info(f"Status: {status}")
    else:
        st.success(f"Status: {status}")

    st.subheader("Caregiver message")
    st.write(caregiver_message)

    st.subheader("Window information")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.write(f"**Subject:** {sample['subject']}")
    info_col2.write(f"**Start second:** {int(sample['start_sec'])}")
    info_col3.write("**Model type:** Random Forest")

    st.subheader("Feature summary")
    feature_groups = {
        "EDA mean": [c for c in feature_columns if c.startswith("EDA") and "mean" in c],
        "BVP std": [c for c in feature_columns if c.startswith("BVP") and "std" in c],
        "TEMP mean": [c for c in feature_columns if c.startswith("TEMP") and "mean" in c],
        "ACC std": [c for c in feature_columns if c.startswith("ACC") and "std" in c],
    }

    rows = []
    for group_name, cols in feature_groups.items():
        for col in cols:
            rows.append({"Feature group": group_name, "Feature": col, "Value": float(sample[col])})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("No summary features found.")

    with st.expander("Important limitations"):
        st.write(
            """
            - This dashboard uses WESAD lab data, not your own MindLink device data yet.
            - The model detects baseline vs stress-pattern changes.
            - It does not detect exact emotions.
            - It does not diagnose anxiety, autism, panic, or any medical condition.
            - Real product accuracy would need testing on new users and your own sensors.
            """
        )


if __name__ == "__main__":
    main()
