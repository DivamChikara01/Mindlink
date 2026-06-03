# MindLink Sensor Ablation Experiment

Put `sensor_ablation_experiment.py` inside your existing `mindlink_ai_starter` folder.

Run:

```bash
python sensor_ablation_experiment.py
```

This tests:

- ALL sensors
- EDA only
- BVP only
- TEMP only
- ACC only
- EDA + BVP
- EDA + BVP + TEMP

For each sensor set, it compares:

1. SVM only
2. Personalized baseline only
3. Combined SVM + baseline

Outputs go into:

```text
results/sensor_ablation_per_subject.csv
results/sensor_ablation_summary.csv
results/sensor_ablation_best_by_system.csv
results/sensor_ablation_report.txt
```

What to look for:

- best `mean_stress_f1`
- best `mean_stress_recall`
- fewest `total_missed_stress_windows`
- fewest weak subjects under 0.70 stress F1

This is a serious research-style experiment because it answers which sensor groups matter most.
