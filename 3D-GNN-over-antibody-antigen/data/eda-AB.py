#!/usr/bin/env python
"""
Exploratory Data Analysis (EDA) for the AB-Bind dataset.

Goals:
- Load AB-Bind_experimental_data.csv robustly.
- Inspect schema (columns, dtypes, missing values).
- Identify key columns: PDB ID, mutation, ΔΔG.
- Compute dataset-level stats (mutant counts, ΔΔG distribution, class balance).
- Compute per-complex stats (n_mutants, ΔΔG mean/median, etc.).
- Generate basic plots for ΔΔG and per-complex distributions.

This is intentionally written to be:
- robust to working directory (uses __file__).
- somewhat agnostic to exact column names (tries to guess PDB / ΔΔG columns).
"""

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# 1. Paths & loading
# ---------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # /data/eda_ab_bind.py -> repo root
AB_BIND_CSV = PROJECT_ROOT / "data" / "external" / "AB-Bind-Database" / "AB-Bind_experimental_data.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
PLOT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_ab_bind(csv_path: Path) -> pd.DataFrame:
    """Load AB-Bind CSV with a reasonable encoding fallback."""
    print(f"[INFO] Loading AB-Bind from: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"AB-Bind CSV not found at: {csv_path}")

    # Try utf-8 first, then fall back to latin1
    encodings = ("utf-8", "latin1", "cp1252")
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"[INFO] Loaded with encoding='{enc}', shape={df.shape}")
            return df
        except UnicodeDecodeError as e:
            print(f"[WARN] Failed to read with encoding='{enc}': {e}")
            last_err = e

    raise RuntimeError(f"Could not decode {csv_path} with {encodings}: {last_err}")


# ---------------------------
# 2. Column discovery helpers
# ---------------------------

def guess_pdb_column(df: pd.DataFrame) -> str:
    """Guess the column that holds the PDB ID."""
    candidates = [c for c in df.columns if "pdb" in c.lower()]
    if not candidates:
        raise ValueError(
            "Could not find a PDB column. "
            "Check df.columns and set PDB_COL manually."
        )
    if len(candidates) > 1:
        print(f"[WARN] Multiple possible PDB columns: {candidates}. Using {candidates[0]}")
    return candidates[0]


def guess_ddg_column(df: pd.DataFrame) -> str:
    """
    Guess the ΔΔG column.

    Heuristics:
    - column name contains 'ddg' (case-insensitive),
    - or contains 'delta' AND 'g',
    - or obviously looks like change in binding free energy.
    """
    lower_cols = {c.lower(): c for c in df.columns}

    # direct ddg
    ddg_candidates = [orig for lower, orig in lower_cols.items() if "ddg" in lower]
    if ddg_candidates:
        if len(ddg_candidates) > 1:
            print(f"[WARN] Multiple ΔΔG-like columns: {ddg_candidates}. Using {ddg_candidates[0]}")
        return ddg_candidates[0]

    # fallback: 'delta' + 'g'
    alt_candidates = [
        orig for lower, orig in lower_cols.items()
        if "delta" in lower and "g" in lower
    ]
    if alt_candidates:
        if len(alt_candidates) > 1:
            print(f"[WARN] Multiple ΔG/ΔΔG-like columns: {alt_candidates}. Using {alt_candidates[0]}")
        return alt_candidates[0]

    raise ValueError(
        "Could not infer ΔΔG column automatically. "
        "Check df.columns and set DDG_COL manually."
    )


# ---------------------------
# 3. Basic EDA
# ---------------------------

def summarize_schema(df: pd.DataFrame) -> None:
    """Print columns, dtypes, and missingness."""
    print("\n[SCHEMA] Columns:")
    print(df.columns.tolist())

    print("\n[SCHEMA] dtypes:")
    print(df.dtypes)

    print("\n[SCHEMA] head():")
    print(df.head())

    # Missing values per column
    missing = df.isna().sum().sort_values(ascending=False)
    print("\n[SCHEMA] Missing values per column:")
    print(missing[missing > 0])


def add_labels(df: pd.DataFrame, ddg_col: str,
               stab_threshold: float = -0.5,
               destab_threshold: float = 0.5) -> pd.DataFrame:
    """
    Add classification labels based on ΔΔG:
    - improved (ΔΔG <= stab_threshold),
    - worsened (ΔΔG >= destab_threshold),
    - near-neutral (otherwise).
    """
    x = df[ddg_col].astype(float)

    df = df.copy()
    df["is_improved"] = (x <= stab_threshold).astype(int)
    df["is_worsened"] = (x >= destab_threshold).astype(int)
    df["is_neutral"] = ((x > stab_threshold) & (x < destab_threshold)).astype(int)

    print("\n[LABELS] Counts:")
    print("Improved (ΔΔG <= {:.1f}): {}".format(stab_threshold, df["is_improved"].sum()))
    print("Worsened (ΔΔG >= {:.1f}): {}".format(destab_threshold, df["is_worsened"].sum()))
    print("Neutral (in between):      {}".format(df["is_neutral"].sum()))

    return df


def ddg_stats(df: pd.DataFrame, ddg_col: str) -> None:
    """Print summary statistics for ΔΔG."""
    x = df[ddg_col].astype(float)

    print(f"\n[ΔΔG] Basic stats for column '{ddg_col}':")
    print(x.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    print("\n[ΔΔG] Extreme values:")
    print("Most stabilizing (lowest ΔΔG):")
    print(
        df.loc[x.nsmallest(5).index, [ddg_col]]
        .join(df.loc[x.nsmallest(5).index].drop(columns=[ddg_col]))
        .head()
    )
    print("\nMost destabilizing (highest ΔΔG):")
    print(
        df.loc[x.nlargest(5).index, [ddg_col]]
        .join(df.loc[x.nlargest(5).index].drop(columns=[ddg_col]))
        .head()
    )


def per_complex_stats(df: pd.DataFrame, pdb_col: str, ddg_col: str) -> pd.DataFrame:
    """
    Group by PDB (complex) and compute:
    - number of mutants
    - mean / median ΔΔG
    - fraction improved / worsened / neutral
    """
    grouped = df.groupby(pdb_col).agg(
        n_mutants=("is_improved", "size"),
        ddg_mean=(ddg_col, "mean"),
        ddg_median=(ddg_col, "median"),
        ddg_std=(ddg_col, "std"),
        frac_improved=("is_improved", "mean"),
        frac_worsened=("is_worsened", "mean"),
        frac_neutral=("is_neutral", "mean"),
    ).reset_index()

    grouped = grouped.sort_values("n_mutants", ascending=False)
    print("\n[COMPLEX] Per-complex stats (top 10 by n_mutants):")
    print(grouped.head(10))

    out_path = OUT_DIR / "ab_bind_per_complex_stats.csv"
    grouped.to_csv(out_path, index=False)
    print(f"[COMPLEX] Saved per-complex stats to {out_path}")

    return grouped


# ---------------------------
# 4. Plots
# ---------------------------

def plot_ddg_hist(df: pd.DataFrame, ddg_col: str) -> None:
    """Histogram + KDE for ΔΔG."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[ddg_col].astype(float), kde=True, bins=50)
    plt.axvline(0.0, color="red", linestyle="--", label="ΔΔG = 0")
    plt.xlabel("ΔΔG (kcal/mol)")
    plt.ylabel("Count")
    plt.title(f"AB-Bind: ΔΔG distribution ({ddg_col})")
    plt.legend()

    out_path = PLOT_DIR / "ab_bind_ddg_hist.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved ΔΔG histogram to {out_path}")


def plot_label_counts(df: pd.DataFrame) -> None:
    """Bar plot of improved / neutral / worsened counts."""
    counts = {
        "Improved": int(df["is_improved"].sum()),
        "Neutral": int(df["is_neutral"].sum()),
        "Worsened": int(df["is_worsened"].sum()),
    }
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values)
    plt.ylabel("Number of mutants")
    plt.title("AB-Bind: mutation effect classes")

    out_path = PLOT_DIR / "ab_bind_label_counts.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved class count barplot to {out_path}")


def plot_ddg_per_complex(df: pd.DataFrame, pdb_col: str, ddg_col: str, top_k: int = 12) -> None:
    """
    Boxplot of ΔΔG per complex, for the top_k complexes by number of mutants.
    """
    counts = df[pdb_col].value_counts().head(top_k).index.tolist()
    df_top = df[df[pdb_col].isin(counts)].copy()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_top, x=pdb_col, y=ddg_col)
    plt.axhline(0.0, color="red", linestyle="--", label="ΔΔG = 0")
    plt.ylabel("ΔΔG (kcal/mol)")
    plt.xlabel("PDB complex")
    plt.title(f"AB-Bind: ΔΔG per complex (top {top_k} by n_mutants)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()

    out_path = PLOT_DIR / "ab_bind_ddg_per_complex.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved per-complex ΔΔG boxplots to {out_path}")


# ---------------------------
# 5. Main
# ---------------------------

def main():
    print(f"[INFO] CWD:         {Path.cwd()}")
    print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[INFO] AB_BIND_CSV:  {AB_BIND_CSV}")

    df = load_ab_bind(AB_BIND_CSV)

    summarize_schema(df)

    # Infer key columns (override manually if needed)
    pdb_col = guess_pdb_column(df)
    print(f"[INFO] Using PDB column: {pdb_col}")

    ddg_col = guess_ddg_column(df)
    print(f"[INFO] Using ΔΔG column: {ddg_col}")

    # Remove rows with missing ΔΔG
    before = df.shape[0]
    df = df.dropna(subset=[ddg_col])
    after = df.shape[0]
    if after < before:
        print(f"[CLEAN] Dropped {before - after} rows with missing ΔΔG.")

    # Coerce ΔΔG to float
    df[ddg_col] = pd.to_numeric(df[ddg_col], errors="coerce")
    n_nan = df[ddg_col].isna().sum()
    if n_nan > 0:
        print(f"[CLEAN] {n_nan} rows had non-numeric ΔΔG and will be dropped.")
        df = df.dropna(subset=[ddg_col])

    print(f"[INFO] Final shape after ΔΔG cleaning: {df.shape}")

    # Add effect labels
    df = add_labels(df, ddg_col=ddg_col, stab_threshold=-0.5, destab_threshold=0.5)

    # Basic ΔΔG stats
    ddg_stats(df, ddg_col=ddg_col)

    # Per-complex stats & export
    per_complex = per_complex_stats(df, pdb_col=pdb_col, ddg_col=ddg_col)

    # Plots
    sns.set(style="whitegrid")
    plot_ddg_hist(df, ddg_col=ddg_col)
    plot_label_counts(df)
    plot_ddg_per_complex(df, pdb_col=pdb_col, ddg_col=ddg_col, top_k=12)

    # Save cleaned version with labels
    cleaned_path = OUT_DIR / "ab_bind_with_labels.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"[INFO] Saved cleaned AB-Bind with labels to {cleaned_path}")


if __name__ == "__main__":
    main()
