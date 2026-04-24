import json

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import FIGURES_DIR, METRICS_DIR, create_directories


def load_summary(path):
    """Load one model summary file."""
    with open(path, "r") as f:
        return json.load(f)


def summary_to_rows(model_name, summary):
    """Convert summary json into rows."""
    rows = []
    for metric, values in summary.items():
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "mean": values["mean"],
                "std": values["std"],
            }
        )
    return rows


def plot_metric_comparison(df, metric, save_path):
    """Plot one metric across models."""
    metric_df = df[df["metric"] == metric].copy()

    plt.figure(figsize=(7, 5))
    plt.bar(metric_df["model"], metric_df["mean"], yerr=metric_df["std"], capsize=5)
    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    create_directories()

    mf_summary = load_summary(METRICS_DIR / "mf_summary.json")
    mf_cbf_summary = load_summary(METRICS_DIR / "mf_cbf_summary.json")
    two_tower_summary = load_summary(METRICS_DIR / "two_tower_summary.json")

    rows = []
    rows.extend(summary_to_rows("MF", mf_summary))
    rows.extend(summary_to_rows("MF+CBF", mf_cbf_summary))
    rows.extend(summary_to_rows("Two-Tower", two_tower_summary))

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(["metric", "model"]).reset_index(drop=True)

    display_df = results_df.copy()
    display_df["mean"] = display_df["mean"].round(4)
    display_df["std"] = display_df["std"].round(4)

    print("\nModel comparison summary:")
    print(display_df)

    csv_path = METRICS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    for metric in results_df["metric"].unique():
        figure_path = FIGURES_DIR / f"{metric}_comparison.png"
        plot_metric_comparison(results_df, metric, figure_path)

    pivot_df = results_df.pivot(index="metric", columns="model", values="mean").round(4)
    print("\nMean metric table:")
    print(pivot_df)

    print(f"\nSaved comparison table to: {csv_path}")
    print(f"Saved plots to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()