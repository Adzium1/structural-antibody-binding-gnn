#!/usr/bin/env python
"""
Process the cleaned AB-Bind table into feature vectors and group splits.

This script aggregates physicochemical shifts for each mutation, exposes
quality metrics from the PDB/assay metadata, and writes:
  * data/processed/ab_bind_features.parquet (features + splits)
  * data/processed/ab_bind_splits.json    (train/val/test row ids)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.structure_utils import (
    StructureCache,
    parse_mutation_list,
    parse_partner_groups,
)

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

SUBPROJECT_ROOT = PROJECT_ROOT / "3D-GNN-over-antibody-antigen"
DEFAULT_INPUT_CSV = (
    SUBPROJECT_ROOT / "data" / "processed" / "ab_bind_with_labels.csv"
)
DEFAULT_FEATURES_PATH = (
    SUBPROJECT_ROOT / "data" / "processed" / "ab_bind_features.csv"
)
DEFAULT_SPLITS_PATH = (
    SUBPROJECT_ROOT / "data" / "processed" / "ab_bind_splits.json"
)

STRUCTURE_CACHE = StructureCache()
STRUCTURAL_NEIGHBOR_RADIUS = 6.0
STRUCTURAL_PARTNER_CUTOFF = 5.0
STRUCTURAL_FEATURES = [
    "mutation_interface_flag",
    "mutation_partner_contact_count",
    "mutation_neighbor_count",
    "mutation_chain_type",
    "mutation_distance_to_partner",
    "mutation_contact_density",
]

MUTATION_PATTERN = re.compile(
    r"(?P<chain>[A-Za-z0-9]+):(?P<wild>[A-Z])(?P<pos>\d+)(?P<mut>[A-Z])"
)

AMINO_ACID_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "polarity": 8.1, "charge": 0},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "polarity": 5.5, "charge": 0},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "polarity": 13.0, "charge": -1},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "polarity": 12.3, "charge": -1},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "polarity": 5.2, "charge": 0},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "polarity": 9.0, "charge": 0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "polarity": 10.4, "charge": 0.1},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "polarity": 5.2, "charge": 0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "polarity": 11.3, "charge": 1},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "polarity": 4.9, "charge": 0},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "polarity": 5.7, "charge": 0},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "polarity": 11.6, "charge": 0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "polarity": 8.0, "charge": 0},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "polarity": 10.5, "charge": 0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "polarity": 10.5, "charge": 1},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "polarity": 9.2, "charge": 0},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "polarity": 8.6, "charge": 0},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "polarity": 5.9, "charge": 0},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "polarity": 5.4, "charge": 0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "polarity": 6.2, "charge": 0},
}


def safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_mutation_entry(entry: str) -> tuple[str, str, float, str] | None:
    if not entry:
        return None
    clean = entry.strip().replace(";", ",")
    match = MUTATION_PATTERN.fullmatch(clean)
    if not match:
        return None
    data = match.groupdict()
    pos = safe_float(data["pos"])
    if pos is None:
        return None
    return data["chain"], data["wild"], pos, data["mut"]


def extract_mutation_features(mutation_text: str) -> Mapping[str, Iterable[float]]:
    chains = set()
    positions = []
    hydro_deltas = []
    volume_deltas = []
    charge_deltas = []
    polarity_deltas = []

    for token in mutation_text.split(","):
        token = token.strip()
        parsed = parse_mutation_entry(token)
        if parsed is None:
            continue
        chain, wild, pos, mutant = parsed
        chains.add(chain)
        positions.append(pos)
        wild_props = AMINO_ACID_PROPERTIES.get(wild)
        mut_props = AMINO_ACID_PROPERTIES.get(mutant)
        if not wild_props or not mut_props:
            continue
        hydro_deltas.append(mut_props["hydrophobicity"] - wild_props["hydrophobicity"])
        volume_deltas.append(mut_props["volume"] - wild_props["volume"])
        charge_deltas.append(mut_props["charge"] - wild_props["charge"])
        polarity_deltas.append(mut_props["polarity"] - wild_props["polarity"])

    return {
        "chains": chains,
        "positions": positions,
        "hydro_deltas": hydro_deltas,
        "volume_deltas": volume_deltas,
        "charge_deltas": charge_deltas,
        "polarity_deltas": polarity_deltas,
    }


def summary_from_list(values: Iterable[float]) -> dict[str, float]:
    seq = np.array(list(values), dtype=float)
    if seq.size == 0:
        return {"mean": 0.0, "std": 0.0, "sum": 0.0}
    return {
        "mean": float(np.mean(seq)),
        "std": float(np.std(seq, ddof=0)),
        "sum": float(np.sum(seq)),
    }


def ensure_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        mutation_text = str(row.get("Mutation", "")).strip()
        if not mutation_text:
            continue
        feature_pack = extract_mutation_features(mutation_text)
        if not feature_pack["positions"]:
            continue
        ddg = safe_float(row.get("ddG(kcal/mol)"))
        if ddg is None:
            continue

        hydro_stats = summary_from_list(feature_pack["hydro_deltas"])
        volume_stats = summary_from_list(feature_pack["volume_deltas"])
        charge_stats = summary_from_list(feature_pack["charge_deltas"])
        polarity_stats = summary_from_list(feature_pack["polarity_deltas"])

        properties = {}
        properties.update(
            {
                f"hydrophobicity_{k}": v
                for k, v in hydro_stats.items()
            }
        )
        properties.update(
            {
                f"volume_{k}": v
                for k, v in volume_stats.items()
            }
        )
        properties.update(
            {
                f"charge_{k}": v
                for k, v in charge_stats.items()
            }
        )
        properties.update(
            {
                f"polarity_{k}": v
                for k, v in polarity_stats.items()
            }
        )

        positions = np.array(feature_pack["positions"], dtype=float)
        position_stats = {
            "position_mean": float(np.mean(positions)),
            "position_std": float(np.std(positions, ddof=0)),
            "mutation_count": len(positions),
            "unique_chain_count": len(feature_pack["chains"]),
        }

        metadata = {
            "pdb_id": row.get("#PDB"),
            "partners": row.get("Partners(A_B)"),
            "mutation": mutation_text,
            "ddg": ddg,
            "is_improved": int(row.get("is_improved", 0)),
            "is_worsened": int(row.get("is_worsened", 0)),
            "is_neutral": int(row.get("is_neutral", 0)),
        }

        for column in (
            "PDB Res. (Angstroms)",
            "PDB R-value",
            "PDB R-free",
            "PDB pH",
            "PDB T (K)",
            "PDB MolProbity clashscore",
            "Assay pH",
            "Assay Temp (Celcius)",
        ):
            metadata[column] = safe_float(row.get(column))

        record = {**metadata, **position_stats, **properties}
        records.append(record)

    features = pd.DataFrame.from_records(records)
    features["sample_id"] = features.index
    return features


def compute_structural_summary(row: Mapping[str, object]) -> dict[str, object]:
    pdb_id = str(row.get("pdb_id", "") or "")
    if not pdb_id:
        return {
            "mutation_interface_flag": False,
            "mutation_partner_contact_count": 0.0,
            "mutation_neighbor_count": 0.0,
            "mutation_chain_type": 0.5,
            "mutation_distance_to_partner": float("nan"),
            "mutation_contact_density": 0.0,
        }

    structure = STRUCTURE_CACHE.load(pdb_id)
    ab_chains, ag_chains = parse_partner_groups(str(row.get("partners", "") or ""))
    mutation_sites = parse_mutation_list(str(row.get("mutation", "") or ""))
    if not mutation_sites:
        return {
            "mutation_interface_flag": False,
            "mutation_partner_contact_count": 0.0,
            "mutation_neighbor_count": 0.0,
            "mutation_chain_type": 0.5,
            "mutation_distance_to_partner": float("nan"),
            "mutation_contact_density": 0.0,
        }

    neighbor_counts: list[float] = []
    contact_counts: list[float] = []
    chain_type_vals: list[float] = []
    distances: list[float] = []
    interface_flags: list[bool] = []

    for chain, _, pos, _ in mutation_sites:
        residue = structure.find_residue(chain, pos)
        if residue is None:
            continue
        cent = residue.centroid
        neighbor_counts.append(structure.count_neighbors(cent, STRUCTURAL_NEIGHBOR_RADIUS))

        if chain in ab_chains:
            partner_chains = ag_chains or ab_chains
            chain_type = 1.0
        elif chain in ag_chains:
            partner_chains = ab_chains or ag_chains
            chain_type = 0.0
        else:
            partner_chains = ab_chains or ag_chains
            chain_type = 0.5

        chain_type_vals.append(chain_type)
        partner_indices = structure.indices_for_chain_set(partner_chains)
        dist_array = structure.distances_to_point(cent)
        contact = 0
        dist_min = float("inf")
        if partner_indices:
            partner_dists = dist_array[partner_indices]
            contact = int(np.sum(partner_dists <= STRUCTURAL_PARTNER_CUTOFF))
            if partner_dists.size:
                dist_min = float(np.min(partner_dists))
        contact_counts.append(contact)
        interface_flags.append(contact > 0)
        distances.append(dist_min)

    if not chain_type_vals:
        return {
            "mutation_interface_flag": False,
            "mutation_partner_contact_count": 0.0,
            "mutation_neighbor_count": 0.0,
            "mutation_chain_type": 0.5,
            "mutation_distance_to_partner": float("nan"),
            "mutation_contact_density": 0.0,
        }

    neighbor_avg = float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
    contact_avg = float(np.mean(contact_counts)) if contact_counts else 0.0
    distance_finite = [d for d in distances if np.isfinite(d)]
    distance_to_partner = float(np.min(distance_finite)) if distance_finite else float("nan")
    contact_density = contact_avg / (neighbor_avg if neighbor_avg > 0 else 1.0)
    return {
        "mutation_interface_flag": any(interface_flags),
        "mutation_partner_contact_count": contact_avg,
        "mutation_neighbor_count": neighbor_avg,
        "mutation_chain_type": float(np.mean(chain_type_vals)),
        "mutation_distance_to_partner": distance_to_partner,
        "mutation_contact_density": contact_density,
    }


def add_structural_features(features: pd.DataFrame) -> pd.DataFrame:
    stats = {col: [] for col in STRUCTURAL_FEATURES}
    for _, row in features.iterrows():
        summary = compute_structural_summary(row)
        for col in STRUCTURAL_FEATURES:
            stats[col].append(summary[col])
    for col, values in stats.items():
        features[col] = values
    return features


def split_groupwise(
    df: pd.DataFrame, group_col: str, seed: int
) -> tuple[list[int], list[int], list[int]]:
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    groups = df[group_col]
    train_idx, rest_idx = next(splitter.split(df, groups=groups))
    rest_df = df.iloc[rest_idx]

    val_test_split = GroupShuffleSplit(
        n_splits=1, train_size=0.5, random_state=seed
    )
    val_rel, test_rel = next(
        val_test_split.split(rest_df, groups=rest_df[group_col])
    )
    val_idx = rest_df.index[val_rel]
    test_idx = rest_df.index[test_rel]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature table + splits from AB-Bind labels."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Cleaned AB-Bind CSV with labels.",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to store feature parquet.",
    )
    parser.add_argument(
        "--splits-out",
        type=Path,
        default=DEFAULT_SPLITS_PATH,
        help="Path to store split metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {args.input_csv}. Run data/eda-AB.py first."
        )

    df = pd.read_csv(args.input_csv)
    ensure_numeric_columns(
        df,
        (
            "PDB Res. (Angstroms)",
            "PDB R-value",
            "PDB R-free",
            "PDB pH",
            "PDB T (K)",
            "PDB MolProbity clashscore",
            "Assay pH",
            "Assay Temp (Celcius)",
        ),
    )

    features = add_structural_features(build_feature_table(df))
    if features.empty:
        raise RuntimeError("No features were generated from the AB-Bind table.")

    train_idx, val_idx, test_idx = split_groupwise(
        features, group_col="pdb_id", seed=args.seed
    )

    features["split"] = "na"
    features.loc[train_idx, "split"] = "train"
    features.loc[val_idx, "split"] = "val"
    features.loc[test_idx, "split"] = "test"

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_out, index=False)
    print(f"[INFO] Saved {len(features)} rows to {args.features_out}")

    splits = {
        "train_ids": [int(idx) for idx in train_idx],
        "val_ids": [int(idx) for idx in val_idx],
        "test_ids": [int(idx) for idx in test_idx],
    }
    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_out.write_text(json.dumps(splits, indent=2))
    print(f"[INFO] Split metadata written to {args.splits_out}")


if __name__ == "__main__":
    main(parse_arguments())
