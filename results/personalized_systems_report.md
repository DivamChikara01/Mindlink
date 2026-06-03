# MindLink Personalized Systems Comparison

## Research question
Can personalized baseline deviation improve wearable stress-pattern alerting compared to a general machine-learning stress classifier?

## Systems compared
1. SVM only
2. Personalized baseline only
3. Combined SVM + baseline

## Validation method
Outer subject-holdout testing. Each subject is tested after training/tuning on the other subjects.

Thresholds and combined weights are tuned on training subjects only.

## Summary results

| system                     |   mean_accuracy |   mean_stress_precision |   mean_stress_recall |   mean_stress_f1 |   total_false_stress_alerts |   total_missed_stress_windows |   weak_subjects_under_0_70_f1 |
|:---------------------------|----------------:|------------------------:|---------------------:|-----------------:|----------------------------:|------------------------------:|------------------------------:|
| Personalized baseline only |        0.915193 |                0.858101 |             0.915691 |         0.876035 |                          52 |                            30 |                             1 |
| Combined SVM + baseline    |        0.914394 |                0.858306 |             0.905667 |         0.870269 |                          49 |                            34 |                             2 |
| SVM only                   |        0.839435 |                0.785493 |             0.859585 |         0.797936 |                         106 |                            51 |                             3 |

## Honest interpretation
The best system is the one with high stress F1 and stress recall, but false alerts and missed stress windows must also be considered.

For MindLink, stress recall is important because missing stress-pattern changes is a major weakness. Too many false alerts can also make caregivers ignore the device.

## Limitations
- WESAD is lab data, not original MindLink wristband data.
- Baseline calibration uses WESAD baseline labels.
- This is not emotion detection.
- This is not medical diagnosis.
- Real-world testing would require ethical approval and consent.
