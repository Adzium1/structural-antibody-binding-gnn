#!/usr/bin/env python
"""
Compare baseline models against the best GNN run and produce a summary table + plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASELINE_CSV = Path("3D-GNN-over-antibody-antigen") / "reports" / "baseline_metrics.csv"
GNN_CSV = Path("3D-GNN-over-antibody-antigen") / "reports" / "gnn_metrics.csv"
OUTPUT_PNG = Path("3D-GNN-over-antibody-antigen") / "reports" / "compare_gnn_baselines.png"


def load_baselines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["split"] == "test"][["model", "mae", "rmse", "r2"]]


def load_best_gnn(path: Path) -> pd.DataFrame:
    gdf = pd.read_csv(path)
    if "is_best_epoch" in gdf.columns and gdf["is_best_epoch"].any():
        best = gdf[gdf["is_best_epoch"]].iloc[-1]
    else:
        best = gdf.sort_values("val_rmse").iloc[0]
    row = pd.DataFrame(
        [
            {
                "model": "gnn",
                "mae": best["test_mae"],
                "rmse": best["test_rmse"],
                "r2": best["test_r2"],
            }
        ]
    )
    return row


def plot_comparison(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.set_index("model")["mae"].plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_ylabel("Test MAE")
    axes[0].set_title("MAE (lower is better)")

    df.set_index("model")["r2"].plot(kind="bar", ax=axes[1], color="#DD8452")
    axes[1].set_ylabel("Test R²")
    axes[1].set_title("R² (higher is better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    baseline_df = load_baselines(args.baseline_csv)
    gnn_df = load_best_gnn(args.gnn_csv)
    combined = pd.concat([baseline_df, gnn_df], ignore_index=True)
    print("\nTest comparison (baselines vs. GNN):")
    print(combined.to_string(index=False))
    plot_comparison(combined, args.output_png)
    print(f"\nSaved comparison plot to {args.output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GNN vs baselines on test set.")
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--gnn-csv", type=Path, default=GNN_CSV)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    main(parser.parse_args())
