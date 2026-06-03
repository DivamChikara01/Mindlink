"""
MindLink Dashboard v2: Personalized Baseline Mode

Run inside your existing mindlink_ai_starter folder:

    pip install streamlit
    streamlit run mindlink_dashboard_v2.py

What this adds:
- Random Forest stress score
- Personalized baseline deviation score
- Main sensor reason for alert
- Time-series trend graph
- Caregiver-friendly color signal
- Top deviating features

Important:
This is a prototype. It detects physiological baseline/stress-pattern changes.
It does NOT detect exact emotions and is NOT a medical device.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from joblib import load

from train_mindlink_model import process_subject


MODEL_PATH = Path("models/mindlink_random_forest.joblib")
COLUMNS_PATH = Path("models/mindlink_feature_columns.joblib")

SENSOR_PREFIXES = ["EDA", "BVP", "TEMP", "ACC"]


def color_signal(stress_score: float, deviation_score: float) -> Tuple[str, str]:
    """
    Caregiver-facing color signal.
    This is not an emotion label. It is a support signal.
    """
    if stress_score >= 80 and deviation_score >= 70:
        return "RED", "Strong stress-pattern + strong personal-baseline deviation"
    if stress_score >= 70 or deviation_score >= 75:
        return "ORANGE", "Possible stress-pattern change or unusual physiology"
    if stress_score >= 45 or deviation_score >= 45:
        return "YELLOW", "Rising activation or uncertain change"
    return "GREEN", "Baseline-like pattern"


def caregiver_message(stress_score: float, deviation_score: float, main_sensor: str) -> str:
    if stress_score >= 80 and deviation_score >= 70:
        return (
            f"Check in calmly. Main signal: {main_sensor}. Reduce stimulation, avoid pressure, "
            "give space, and offer one simple choice."
        )
    if stress_score >= 70 or deviation_score >= 75:
        return (
            f"Monitor closely. Main signal: {main_sensor}. Look for triggers like noise, crowding, "
            "heat, recent movement, or social pressure."
        )
    if stress_score >= 45 or deviation_score >= 45:
        return (
            f"Possible early change. Main signal: {main_sensor}. Keep observing and compare with behavior."
        )
    return "No alert. Pattern looks close to baseline for this selected window."


def confidence_label(stress_score: float, deviation_score: float) -> str:
    """
    Confidence is higher when stress score and baseline deviation agree.
    """
    stress_high = stress_score >= 70
    dev_high = deviation_score >= 70
    stress_low = stress_score < 40
    dev_low = deviation_score < 40

    if (stress_high and dev_high) or (stress_low and dev_low):
        return "High"
    if abs(stress_score - deviation_score) <= 25:
        return "Medium"
    return "Low / mixed"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        st.error("Model files not found. First run: python train_mindlink_model.py")
        st.stop()

    model = load(MODEL_PATH)
    feature_columns = load(COLUMNS_PATH)
    return model, feature_columns


def robust_baseline_stats(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Compute median and robust scale using baseline windows only.
    Uses MAD, with fallback to standard deviation if MAD is too small.
    """
    baseline_df = df[df["label"] == 0].copy()

    if baseline_df.empty:
        baseline_df = df.copy()

    med = baseline_df[feature_columns].median()
    mad = (baseline_df[feature_columns] - med).abs().median()
    robust_scale = 1.4826 * mad

    std = baseline_df[feature_columns].std().fillna(0)

    scale = robust_scale.copy()
    scale[scale < 1e-8] = std[scale < 1e-8]
    scale[scale < 1e-8] = 1.0

    return med, scale


def add_predictions_and_deviation(
    df: pd.DataFrame,
    model,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Add:
    - stress_score from Random Forest
    - personalized baseline deviation score
    - sensor-level deviation scores
    - main sensor reason
    - color signal
    """
    out = df.copy()
    X = out[feature_columns]

    out["prediction"] = model.predict(X)
    out["stress_score"] = model.predict_proba(X)[:, 1] * 100

    med, scale = robust_baseline_stats(out, feature_columns)
    z = ((out[feature_columns] - med).abs() / scale).clip(upper=12)

    sensor_cols: Dict[str, List[str]] = {
        sensor: [c for c in feature_columns if c.startswith(sensor)]
        for sensor in SENSOR_PREFIXES
    }

    for sensor, cols in sensor_cols.items():
        if cols:
            out[f"{sensor}_deviation_z"] = z[cols].median(axis=1)
        else:
            out[f"{sensor}_deviation_z"] = 0.0

    sensor_z_cols = [f"{sensor}_deviation_z" for sensor in SENSOR_PREFIXES]
    out["max_deviation_z"] = out[sensor_z_cols].max(axis=1)

    # Convert z-score to 0-100 score. Higher z = farther from personal baseline.
    out["baseline_deviation_score"] = out["max_deviation_z"].apply(
        lambda v: min(100.0, 100.0 * (1.0 - math.exp(-float(v) / 3.0)))
    )

    def main_sensor(row) -> str:
        values = {sensor: row[f"{sensor}_deviation_z"] for sensor in SENSOR_PREFIXES}
        return max(values, key=values.get)

    out["main_sensor"] = out.apply(main_sensor, axis=1)
    out["color_signal"] = out.apply(
        lambda row: color_signal(row["stress_score"], row["baseline_deviation_score"])[0],
        axis=1,
    )

    return out


@st.cache_data
def load_subject_df(subject_path: str) -> pd.DataFrame:
    rows = process_subject(subject_path)
    return pd.DataFrame(rows).dropna()


def top_deviating_features(
    df_scored: pd.DataFrame,
    sample_index,
    feature_columns: List[str],
    top_n: int = 12,
) -> pd.DataFrame:
    med, scale = robust_baseline_stats(df_scored, feature_columns)
    sample = df_scored.loc[sample_index, feature_columns]
    z = ((sample - med).abs() / scale).sort_values(ascending=False)

    rows = []
    for feature, value in z.head(top_n).items():
        rows.append(
            {
                "Feature": feature,
                "Robust deviation z": round(float(value), 3),
                "Current value": round(float(sample[feature]), 5),
                "Baseline median": round(float(med[feature]), 5),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="MindLink Personalized Baseline Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    st.title("MindLink Personalized Baseline Dashboard v2")
    st.caption(
        "Combines a trained stress-pattern model with person-specific baseline deviation."
    )

    model, feature_columns = load_model()
    subject_files = sorted(glob.glob("data/WESAD/S*/S*.pkl"))

    if not subject_files:
        st.error("No WESAD files found. Expected: data/WESAD/S2/S2.pkl")
        st.stop()

    with st.sidebar:
        st.header("Controls")
        subject_path = st.selectbox("Choose subject", subject_files)
        raw_df = load_subject_df(subject_path)
        df_scored = add_predictions_and_deviation(raw_df, model, feature_columns)

        mode = st.selectbox(
            "View mode",
            ["All windows", "True baseline only", "True stress only", "High alert windows"]
        )

        if mode == "True baseline only":
            shown_df = df_scored[df_scored["label"] == 0].copy()
        elif mode == "True stress only":
            shown_df = df_scored[df_scored["label"] == 1].copy()
        elif mode == "High alert windows":
            shown_df = df_scored[
                (df_scored["stress_score"] >= 70) |
                (df_scored["baseline_deviation_score"] >= 75)
            ].copy()
        else:
            shown_df = df_scored.copy()

        if shown_df.empty:
            st.warning("No windows match this mode.")
            st.stop()

        selected_position = st.slider(
            "Window number",
            min_value=0,
            max_value=len(shown_df) - 1,
            value=0,
        )

        st.divider()
        st.write("Not emotion detection. Not medical diagnosis.")

    sample = shown_df.iloc[selected_position]
    sample_index = sample.name

    stress_score = float(sample["stress_score"])
    deviation_score = float(sample["baseline_deviation_score"])
    color, color_reason = color_signal(stress_score, deviation_score)
    confidence = confidence_label(stress_score, deviation_score)
    main_sensor = str(sample["main_sensor"])

    true_label = "Stress" if int(sample["label"]) == 1 else "Baseline"
    pred_label = "Stress-pattern change" if int(sample["prediction"]) == 1 else "Baseline"

    tab_live, tab_trends, tab_explain, tab_limits = st.tabs(
        ["Live window", "Trends", "Why it alerted", "Limitations"]
    )

    with tab_live:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Color signal", color)
        col2.metric("Stress score", f"{stress_score:.2f}%")
        col3.metric("Baseline deviation", f"{deviation_score:.2f}%")
        col4.metric("Confidence", confidence)
        col5.metric("Main signal", main_sensor)

        st.progress(min(max(stress_score / 100, 0), 1), text="Stress-pattern score")
        st.progress(min(max(deviation_score / 100, 0), 1), text="Personal baseline deviation")

        if color == "RED":
            st.error(f"{color}: {color_reason}")
        elif color == "ORANGE":
            st.warning(f"{color}: {color_reason}")
        elif color == "YELLOW":
            st.info(f"{color}: {color_reason}")
        else:
            st.success(f"{color}: {color_reason}")

        st.subheader("Caregiver message")
        st.write(caregiver_message(stress_score, deviation_score, main_sensor))

        st.subheader("Window information")
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**Subject:** {sample['subject']}")
        c2.write(f"**Start second:** {int(sample['start_sec'])}")
        c3.write(f"**True WESAD label:** {true_label}")
        c4.write(f"**Model prediction:** {pred_label}")

        st.subheader("Sensor deviation from this subject's baseline")
        sensor_table = pd.DataFrame(
            [
                {"Sensor": "EDA", "Deviation z": round(float(sample["EDA_deviation_z"]), 3)},
                {"Sensor": "BVP", "Deviation z": round(float(sample["BVP_deviation_z"]), 3)},
                {"Sensor": "TEMP", "Deviation z": round(float(sample["TEMP_deviation_z"]), 3)},
                {"Sensor": "ACC", "Deviation z": round(float(sample["ACC_deviation_z"]), 3)},
            ]
        )
        st.dataframe(sensor_table, use_container_width=True)

    with tab_trends:
        st.subheader("Stress and baseline deviation over time")

        chart_df = df_scored[[
            "start_sec",
            "stress_score",
            "baseline_deviation_score",
        ]].copy()
        chart_df = chart_df.rename(
            columns={
                "start_sec": "Start second",
                "stress_score": "Stress score",
                "baseline_deviation_score": "Baseline deviation score",
            } 
        )
        st.line_chart(chart_df.set_index("Start second"))

        st.subheader("Color signal counts")
        st.dataframe(
            df_scored["color_signal"].value_counts().rename_axis("Color").reset_index(name="Windows"),
            use_container_width=True,
        )

        st.subheader("Highest-alert windows")
        high = df_scored.sort_values(
            ["stress_score", "baseline_deviation_score"],
            ascending=False
        )[[
            "start_sec",
            "label",
            "prediction",
            "stress_score",
            "baseline_deviation_score",
            "main_sensor",
            "color_signal",
        ]].head(15)
        st.dataframe(high, use_container_width=True)

    with tab_explain:
        st.subheader("Top features farthest from personal baseline")
        st.write(
            "These are not proof of emotion. They show which feature values moved farthest from this subject's baseline."
        )
        st.dataframe(
            top_deviating_features(df_scored, sample_index, feature_columns),
            use_container_width=True,
        )

        st.subheader("How this v2 system works")
        st.markdown(
            """
            1. The Random Forest gives a **stress-pattern score** using WESAD-trained features.
            2. The personalized baseline module calculates this subject's baseline from baseline-labeled windows.
            3. Each new window is compared against that subject's baseline using robust z-scores.
            4. The dashboard combines stress score + baseline deviation into a simple caregiver-facing color signal.
            """
        )

    with tab_limits:
        st.subheader("Brutal limitations")
        st.markdown(
            """
            - This still uses WESAD lab data, not MindLink's own wristband data.
            - It does not know exact emotions.
            - It can confuse stress with heat, movement, caffeine, excitement, or sensor noise.
            - The model was trained on a small public dataset.
            - A real product needs original sensor data, user feedback, comfort testing, and human-subject ethics review.
            """
        )


if __name__ == "__main__":
    main()
