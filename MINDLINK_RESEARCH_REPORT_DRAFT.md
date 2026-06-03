# MindLink Research Report Draft

## Title
MindLink: Personalized Baseline Deviation for Wearable Stress-Pattern Detection

## Abstract
MindLink is a wearable AI caregiver-support concept designed to detect physiological stress-pattern changes relative to an individual's baseline. The current prototype uses WESAD wrist-sensor data and compares general machine-learning classification with personalized baseline deviation. In subject-holdout testing, personalized baseline methods outperformed a general SVM-only classifier. The strongest sensor-ablation result used **EDA_BVP_TEMP** with **Combined SVM + baseline**, reaching **0.901 mean stress F1** and **0.947 mean stress recall**. These results support the idea that stress-pattern alerting should be personalized rather than based only on a one-size-fits-all model.

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
| system                     |   mean_accuracy |   mean_stress_precision |   mean_stress_recall |   mean_stress_f1 |   total_false_stress_alerts |   total_missed_stress_windows |   weak_subjects_under_0_70_f1 |
|:---------------------------|----------------:|------------------------:|---------------------:|-----------------:|----------------------------:|------------------------------:|------------------------------:|
| Personalized baseline only |        0.915193 |                0.858101 |             0.915691 |         0.876035 |                          52 |                            30 |                             1 |
| Combined SVM + baseline    |        0.914394 |                0.858306 |             0.905667 |         0.870269 |                          49 |                            34 |                             2 |
| SVM only                   |        0.839435 |                0.785493 |             0.859585 |         0.797936 |                         106 |                            51 |                             3 |

### Interpretation
The best system in this experiment was **Personalized baseline only**, with **0.876 mean stress F1** and **0.916 mean stress recall**. It outperformed the SVM-only baseline, which reached **0.798 mean stress F1**.

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
| sensor_set   | system                     |   mean_accuracy |   mean_stress_precision |   mean_stress_recall |   mean_stress_f1 |   total_false_stress_alerts |   total_missed_stress_windows |   weak_subjects_under_0_70_f1 |
|:-------------|:---------------------------|----------------:|------------------------:|---------------------:|-----------------:|----------------------------:|------------------------------:|------------------------------:|
| EDA_BVP_TEMP | Combined SVM + baseline    |        0.928641 |                0.867632 |             0.947101 |         0.901406 |                          50 |                            19 |                             1 |
| ALL          | Personalized baseline only |        0.915193 |                0.858101 |             0.915691 |         0.876035 |                          52 |                            30 |                             1 |
| EDA_BVP_TEMP | Personalized baseline only |        0.915193 |                0.858101 |             0.915691 |         0.876035 |                          52 |                            30 |                             1 |
| ALL          | Combined SVM + baseline    |        0.914394 |                0.858306 |             0.905667 |         0.870269 |                          49 |                            34 |                             2 |
| ALL          | SVM only                   |        0.839435 |                0.785493 |             0.859585 |         0.797936 |                         106 |                            51 |                             3 |
| ACC_only     | Combined SVM + baseline    |        0.813379 |                0.728373 |             0.815407 |         0.750049 |                         115 |                            68 |                             4 |
| EDA_only     | Personalized baseline only |        0.850976 |                0.775327 |             0.787598 |         0.747644 |                          67 |                            79 |                             4 |
| BVP_only     | Combined SVM + baseline    |        0.784365 |                0.692329 |             0.836674 |         0.745028 |                         151 |                            60 |                             6 |
| EDA_only     | Combined SVM + baseline    |        0.845925 |                0.774106 |             0.773709 |         0.739941 |                          67 |                            84 |                             4 |
| EDA_BVP_TEMP | SVM only                   |        0.792876 |                0.693695 |             0.81594  |         0.733096 |                         135 |                            68 |                             5 |

### Interpretation
The strongest sensor-ablation result used **EDA_BVP_TEMP** with **Combined SVM + baseline**. This suggests that selected physiological signals may outperform using all sensors indiscriminately.

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
MindLink's current results support a personalized baseline approach for wearable stress-pattern alerting. The best system achieved **0.901 mean stress F1** and **0.947 mean stress recall** using **EDA_BVP_TEMP**. The project is not yet a product or medical device, but it is now a legitimate research prototype with a clear experimental direction.
