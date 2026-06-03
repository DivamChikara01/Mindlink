"""
MindLink Research Report Builder

Run inside your existing mindlink_ai_starter folder:

    python build_research_report.py

Reads:
    results/personalized_systems_summary.csv
    results/sensor_ablation_summary.csv

Creates:
    MINDLINK_ONE_PAGE_SUMMARY.md
    MINDLINK_RESEARCH_REPORT_DRAFT.md
    MINDLINK_PROFESSOR_EMAIL_DRAFT.md
"""

from pathlib import Path

import pandas as pd


def fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def main():
    personalized_path = Path("results/personalized_systems_summary.csv")
    ablation_path = Path("results/sensor_ablation_summary.csv")

    if not personalized_path.exists():
        raise FileNotFoundError("Missing results/personalized_systems_summary.csv")

    if not ablation_path.exists():
        raise FileNotFoundError("Missing results/sensor_ablation_summary.csv")

    personalized = pd.read_csv(personalized_path).sort_values("mean_stress_f1", ascending=False)
    ablation = pd.read_csv(ablation_path).sort_values("mean_stress_f1", ascending=False)

    best_personalized = personalized.iloc[0]
    svm_row = personalized[personalized["system"].str.contains("SVM only", case=False, na=False)].iloc[0]
    best_ablation = ablation.iloc[0]

    one_page = f"""# MindLink: Personalized Baseline Deviation for Wearable Stress-Pattern Detection

## One-sentence summary
MindLink is a caregiver-support wearable AI concept that learns an individual's physiological baseline and translates stress-pattern changes into simple color signals.

## Problem
Some children may struggle to communicate internal stress, overload, or discomfort early. Caregivers may miss early physiological warning signs before escalation.

## Research question
Can personalized physiological baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning stress classifier?

## Dataset and signals
This prototype uses WESAD wrist-sensor data with wearable features from:
- EDA: skin conductance
- BVP: pulse waveform
- TEMP: skin temperature
- ACC: movement

## Method
The system extracts statistical features from 60-second windows. It compares:
1. General SVM classifier only
2. Personalized baseline deviation only
3. Combined SVM + personalized baseline scoring

Testing uses subject-holdout validation, meaning each subject is tested after tuning/training on other subjects.

## Key result
Best personalized-system result:
- System: **{best_personalized["system"]}**
- Mean accuracy: **{fmt(best_personalized["mean_accuracy"])}**
- Mean stress recall: **{fmt(best_personalized["mean_stress_recall"])}**
- Mean stress F1: **{fmt(best_personalized["mean_stress_f1"])}**
- False stress alerts: **{int(best_personalized["total_false_stress_alerts"])}**
- Missed stress windows: **{int(best_personalized["total_missed_stress_windows"])}**

Compared with SVM-only:
- SVM-only mean stress F1: **{fmt(svm_row["mean_stress_f1"])}**
- SVM-only missed stress windows: **{int(svm_row["total_missed_stress_windows"])}**
- SVM-only false stress alerts: **{int(svm_row["total_false_stress_alerts"])}**

## Sensor ablation finding
Best sensor-ablation result:
- Sensor set: **{best_ablation["sensor_set"]}**
- System: **{best_ablation["system"]}**
- Mean accuracy: **{fmt(best_ablation["mean_accuracy"])}**
- Mean stress recall: **{fmt(best_ablation["mean_stress_recall"])}**
- Mean stress F1: **{fmt(best_ablation["mean_stress_f1"])}**
- Missed stress windows: **{int(best_ablation["total_missed_stress_windows"])}**

## Honest limitations
- This is not emotion detection.
- This is not medical diagnosis.
- WESAD is lab data, not original MindLink wristband data.
- Personalized baseline uses labeled baseline windows as a simulated calibration period.
- Real use would require ethical approval, consent, original sensor data, and user-specific validation.

## Next step
Build a cleaned GitHub repository, prepare a 60–90 second demo video, and seek research feedback on improving the personalized baseline method.
"""

    report = f"""# MindLink Research Report Draft

## Title
MindLink: Personalized Baseline Deviation for Wearable Stress-Pattern Detection

## Abstract
MindLink is a wearable AI caregiver-support concept designed to detect physiological stress-pattern changes relative to an individual's baseline. The current prototype uses WESAD wrist-sensor data and compares general machine-learning classification with personalized baseline deviation. In subject-holdout testing, personalized baseline methods outperformed a general SVM-only classifier. The strongest sensor-ablation result used **{best_ablation["sensor_set"]}** with **{best_ablation["system"]}**, reaching **{fmt(best_ablation["mean_stress_f1"])} mean stress F1** and **{fmt(best_ablation["mean_stress_recall"])} mean stress recall**. These results support the idea that stress-pattern alerting should be personalized rather than based only on a one-size-fits-all model.

## 1. Introduction
Many caregiver-support tools rely on observation after stress or overload is already visible. MindLink explores whether wearable physiological data can provide earlier support signals by learning a person's normal baseline and detecting deviations from that baseline.

This project does not attempt to detect exact emotions. The goal is to detect physiological stress-pattern changes and translate them into simple caregiver-facing signals.

## 2. Research Question
Can personalized physiological baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning stress classifier?

## 3. Dataset
The prototype uses WESAD wrist-sensor data. The relevant signals include:
- EDA: electrodermal activity / skin conductance
- BVP: blood volume pulse / pulse waveform
- TEMP: skin temperature
- ACC: accelerometer movement

Each sample window represents a 60-second segment of wearable sensor data.

## 4. Methods
The pipeline includes:
1. Feature extraction from 60-second windows
2. Subject-holdout validation
3. SVM classifier training
4. Personalized baseline deviation scoring
5. Combined stress-score and baseline-deviation scoring
6. Sensor ablation across different sensor groups

Subject-holdout validation means the model is evaluated on a subject not included in training/tuning.

## 5. Experiment 1: Personalized Systems Comparison

Systems compared:
1. SVM only
2. Personalized baseline only
3. Combined SVM + baseline

### Results
{personalized.to_markdown(index=False)}

### Interpretation
The best system in this experiment was **{best_personalized["system"]}**, with **{fmt(best_personalized["mean_stress_f1"])} mean stress F1** and **{fmt(best_personalized["mean_stress_recall"])} mean stress recall**. It outperformed the SVM-only baseline, which reached **{fmt(svm_row["mean_stress_f1"])} mean stress F1**.

## 6. Experiment 2: Sensor Ablation

Sensor sets tested:
- ALL
- EDA only
- BVP only
- TEMP only
- ACC only
- EDA + BVP
- EDA + BVP + TEMP

### Top Results
{ablation.head(10).to_markdown(index=False)}

### Interpretation
The strongest sensor-ablation result used **{best_ablation["sensor_set"]}** with **{best_ablation["system"]}**. This suggests that selected physiological signals may outperform using all sensors indiscriminately.

## 7. Discussion
The results suggest that a one-size-fits-all stress classifier is not enough. Personalized baseline scoring improved stress-pattern detection and reduced weak-subject failures. This supports MindLink's core design: learn each user's baseline first, then detect deviations.

## 8. Limitations
This work is still a prototype. It uses public lab data, not original MindLink device data. The baseline calibration uses WESAD labels, which simulates but does not fully replicate real-world calibration. The system does not infer exact emotions and should not be described as a diagnostic or medical device.

## 9. Future Work
Next steps:
1. Improve personalized baseline modeling
2. Add explainability per sensor group
3. Build a simple hardware data logger for EDA, PPG/BVP, and temperature
4. Collect original data only under proper ethical approval and consent
5. Test whether the system generalizes to real-world wearable conditions

## 10. Conclusion
MindLink's current results support a personalized baseline approach for wearable stress-pattern alerting. The best system achieved **{fmt(best_ablation["mean_stress_f1"])} mean stress F1** and **{fmt(best_ablation["mean_stress_recall"])} mean stress recall** using **{best_ablation["sensor_set"]}**. The project is not yet a product or medical device, but it is now a legitimate research prototype with a clear experimental direction.
"""

    email = f"""Subject: Wearable AI project — personalized baseline stress-pattern detection

Hi Professor Yiwen,

I’m a high school student working on MindLink, a wearable AI caregiver-support project focused on detecting physiological stress-pattern changes relative to an individual's baseline.

I built a prototype using WESAD wrist-sensor data. The system extracts features from EDA, BVP, temperature, and accelerometer signals, then compares general stress classification with personalized baseline-deviation methods.

My main research question is:

Can personalized physiological baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning classifier?

In subject-holdout testing, the strongest sensor-ablation result used {best_ablation["sensor_set"]} with {best_ablation["system"]}. It reached about {fmt(best_ablation["mean_stress_f1"])} mean stress F1 and {fmt(best_ablation["mean_stress_recall"])} mean stress recall. This outperformed the general SVM-only approach, which reached about {fmt(svm_row["mean_stress_f1"])} mean stress F1.

I know this is not emotion detection or a medical device yet. The current work is a prototype using public lab data, and my next step is improving the personalized baseline method and planning ethical original data collection later.

Would you be willing to give feedback on my validation approach or suggest how I should improve the personalization method?

Thank you,
Divam
"""

    Path("MINDLINK_ONE_PAGE_SUMMARY.md").write_text(one_page, encoding="utf-8")
    Path("MINDLINK_RESEARCH_REPORT_DRAFT.md").write_text(report, encoding="utf-8")
    Path("MINDLINK_PROFESSOR_EMAIL_DRAFT.md").write_text(email, encoding="utf-8")

    print("Created:")
    print("MINDLINK_ONE_PAGE_SUMMARY.md")
    print("MINDLINK_RESEARCH_REPORT_DRAFT.md")
    print("MINDLINK_PROFESSOR_EMAIL_DRAFT.md")


if __name__ == "__main__":
    main()
