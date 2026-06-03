"""
MindLink Research Figures Generator

Run inside your existing mindlink_ai_starter folder:

    python make_research_figures.py

Reads:
    results/personalized_systems_summary.csv
    results/sensor_ablation_summary.csv

Creates:
    results/figures/system_stress_f1.png
    results/figures/system_stress_recall.png
    results/figures/system_missed_stress.png
    results/figures/sensor_ablation_top_f1.png
    results/figures/sensor_ablation_missed_stress.png
    results/figures/sensor_ablation_false_alerts.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_chart(df, x_col, y_col, title, ylabel, filename, rotation=25, top_n=None):
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def main():
    personalized_path = Path("results/personalized_systems_summary.csv")
    ablation_path = Path("results/sensor_ablation_summary.csv")

    if not personalized_path.exists():
        raise FileNotFoundError("Missing results/personalized_systems_summary.csv")

    if not ablation_path.exists():
        raise FileNotFoundError("Missing results/sensor_ablation_summary.csv")

    personalized = pd.read_csv(personalized_path)
    ablation = pd.read_csv(ablation_path)

    # System comparison charts.
    system_order = personalized.sort_values("mean_stress_f1", ascending=False)

    save_bar_chart(
        system_order,
        "system",
        "mean_stress_f1",
        "MindLink System Comparison: Mean Stress F1",
        "Mean stress F1",
        "system_stress_f1.png",
    )

    save_bar_chart(
        system_order.sort_values("mean_stress_recall", ascending=False),
        "system",
        "mean_stress_recall",
        "MindLink System Comparison: Mean Stress Recall",
        "Mean stress recall",
        "system_stress_recall.png",
    )

    save_bar_chart(
        system_order.sort_values("total_missed_stress_windows", ascending=True),
        "system",
        "total_missed_stress_windows",
        "MindLink System Comparison: Missed Stress Windows",
        "Total missed stress windows",
        "system_missed_stress.png",
    )

    # Sensor ablation charts.
    ablation = ablation.copy()
    ablation["label"] = ablation["sensor_set"] + " — " + ablation["system"]

    top_f1 = ablation.sort_values("mean_stress_f1", ascending=False)

    save_bar_chart(
        top_f1,
        "label",
        "mean_stress_f1",
        "Sensor Ablation: Top Mean Stress F1 Results",
        "Mean stress F1",
        "sensor_ablation_top_f1.png",
        rotation=35,
        top_n=12,
    )

    save_bar_chart(
        ablation.sort_values("total_missed_stress_windows", ascending=True),
        "label",
        "total_missed_stress_windows",
        "Sensor Ablation: Fewest Missed Stress Windows",
        "Total missed stress windows",
        "sensor_ablation_missed_stress.png",
        rotation=35,
        top_n=12,
    )

    save_bar_chart(
        ablation.sort_values("total_false_stress_alerts", ascending=True),
        "label",
        "total_false_stress_alerts",
        "Sensor Ablation: Fewest False Stress Alerts",
        "Total false stress alerts",
        "sensor_ablation_false_alerts.png",
        rotation=35,
        top_n=12,
    )

    print("Saved figures to:", FIG_DIR)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
