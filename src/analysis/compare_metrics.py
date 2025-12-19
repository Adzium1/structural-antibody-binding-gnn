#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASELINE_CSV = (
    Path("3D-GNN-over-antibody-antigen")
    / "reports"
    / "baseline_metrics.csv"
)
GNN_METRICS_CSV = (
    Path("3D-GNN-over-antibody-antigen")
    / "reports"
    / "gnn_metrics.csv"
)
OUTPUT_PLOT = (
    Path("3D-GNN-over-antibody-antigen")
    / "reports"
    / "comparison_plot.png"
)


def main(args: argparse.Namespace) -> None:
    baseline = pd.read_csv(BASELINE_CSV)
    gnn_metrics = pd.read_csv(GNN_METRICS_CSV)
    best_rows = gnn_metrics[gnn_metrics["is_best_epoch"]]
    if best_rows.empty:
        raise ValueError("No best epoch found in gnn metrics.")
    best = best_rows.iloc[-1]

    comparison = (
        baseline[baseline["split"] == "test"]
        .loc[:, ["model", "mae", "rmse", "r2"]]
        .copy()
    )
    gnn_row = pd.DataFrame(
        [
            {
                "model": "gnn",
                "mae": best["test_mae"],
                "rmse": best["test_rmse"],
                "r2": best["test_r2"],
            }
        ]
    )
    comparison = pd.concat([comparison, gnn_row], ignore_index=True)

    print("\nTest-set comparison:")
    print(comparison.to_string(index=False))

    comparison = comparison.set_index("model")

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison["mae"].plot(kind="bar", ax=ax, color="#4C72B0", label="MAE")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_xlabel("Model")
    ax.set_title("Baseline vs. GNN test metrics")
    ax2 = ax.twinx()
    comparison["r2"].plot(
        kind="line",
        marker="o",
        lw=2,
        color="#DD8452",
        ax=ax2,
        label="Test R²",
    )
    ax2.set_ylabel("Test R²")
    ax2.grid(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nSaved comparison plot to {OUTPUT_PLOT}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs. GNN metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
