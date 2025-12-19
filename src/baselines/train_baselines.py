#!/usr/bin/env python
"""
Train simple baselines (linear + tree) on the AB-Bind feature table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

SUBPROJECT_ROOT = PROJECT_ROOT / "3D-GNN-over-antibody-antigen"
DEFAULT_FEATURES = (
    SUBPROJECT_ROOT / "data" / "processed" / "ab_bind_features.csv"
)
DEFAULT_METRICS = SUBPROJECT_ROOT / "reports" / "baseline_metrics.csv"

EXCLUDED_NUMERIC = {
    "sample_id",
    "ddg",
    "is_improved",
    "is_worsened",
    "is_neutral",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit baseline regressors on AB-Bind features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Parquet file created by src/data/prepare_ab_bind.py.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS,
        help="Where to write the evaluation metrics.",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for randomized algorithms."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of threads for tree models."
    )
    return parser.parse_args()


def build_model_pool(seed: int, jobs: int) -> Dict[str, object]:
    pool: Dict[str, object] = {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=jobs
        ),
        "gbt": GradientBoostingRegressor(
            learning_rate=0.05, n_estimators=200, random_state=seed
        ),
    }
    if XGBRegressor is not None:
        pool["xgboost"] = XGBRegressor(
            objective="reg:squarederror", random_state=seed, n_jobs=jobs, verbosity=0
        )
    return pool


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    label = df["ddg"].to_numpy(dtype=float)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in EXCLUDED_NUMERIC]
    X = df[feature_cols].to_numpy(dtype=float)
    return X, label, df["split"].to_numpy(dtype=str), feature_cols


def split_data(
    X: np.ndarray, y: np.ndarray, splits: np.ndarray
) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    masks = {name: splits == name for name in ("train", "val", "test")}
    return {
        name: (X[mask], y[mask])
        for name, mask in masks.items()
    }


def evaluate_model(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    preds = pipeline.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main(args: argparse.Namespace) -> None:
    if not args.features.exists():
        raise FileNotFoundError("Run src/data/prepare_ab_bind.py first.")

    features = pd.read_csv(args.features)
    X, y, splits, feature_cols = prepare_features(features)
    data_slices = split_data(X, y, splits)

    model_pool = build_model_pool(args.random_state, args.n_jobs)
    metrics = []

    for name, estimator in model_pool.items():
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        xt, yt = data_slices["train"]
        pipeline.fit(xt, yt)
        for split_name, (xs, ys) in data_slices.items():
            if xs.size == 0:
                continue
            stats = evaluate_model(pipeline, xs, ys)
            metrics.append(
                {
                    "model": name,
                    "split": split_name,
                    "mae": stats["mae"],
                    "rmse": stats["rmse"],
                    "r2": stats["r2"],
                }
            )
            print(
                f"[{name}/{split_name}] MAE={stats['mae']:.3f} "
                f"RMSE={stats['rmse']:.3f} R2={stats['r2']:.3f}"
            )

    metrics_df = pd.DataFrame(metrics)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)
    print(f"[INFO] Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main(parse_arguments())
