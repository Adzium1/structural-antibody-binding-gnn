#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.structure_utils import (
    StructureCache,
    parse_mutation_list,
    parse_partner_groups,
    STANDARD_AMINO_ACIDS,
    SUBPROJECT_ROOT,
)

FEATURES_PATH = (
    SUBPROJECT_ROOT / "data" / "processed" / "ab_bind_features.csv"
)
OUTPUT_DIR = SUBPROJECT_ROOT / "data" / "graphs"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "ab_bind_graphs.pkl"

GRAPH_RADIUS = 10.0
EDGE_RADIUS = 8.0
INTERFACE_THRESHOLD = 5.0
SOLVENT_RADIUS = 5.0

# A small set of engineered features (from ab_bind_features.csv) to inject into node vectors.
# These were useful for the tabular baselines; we append the same vector to every node
# of a graph so the model has access to global mutation-level context.
ENGINEERED_COLUMNS = [
    "hydrophobicity_mean",
    "hydrophobicity_std",
    "volume_mean",
    "volume_std",
    "charge_mean",
    "charge_std",
    "polarity_mean",
    "polarity_std",
    "mutation_count",
    "unique_chain_count",
]


def extract_engineered_features(row: pd.Series) -> np.ndarray:
    values = []
    for col in ENGINEERED_COLUMNS:
        val = row.get(col, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        values.append(val)
    return np.array(values, dtype=float)


def build_graph(
    row: pd.Series,
    structure_cache: StructureCache,
    graph_radius: float,
    edge_radius: float,
    interface_threshold: float,
) -> dict | None:
    pdb_id = str(row["pdb_id"])
    try:
        structure = structure_cache.load(pdb_id)
    except FileNotFoundError:
        return None

    mutation_sites = parse_mutation_list(str(row["mutation"]))
    if not mutation_sites:
        return None

    centers = []
    mutated_keys = set()
    for chain, _, pos, _ in mutation_sites:
        residue = structure.find_residue(chain, pos)
        if residue is None:
            continue
        centers.append(residue.centroid)
        mutated_keys.add((residue.chain_id, residue.resseq))

    if not centers:
        return None

    center_point = np.mean(np.stack(centers), axis=0)
    distances = structure.distances_to_point(center_point)
    node_mask = distances <= graph_radius
    if not np.any(node_mask):
        return None

    node_indices = np.where(node_mask)[0]
    ab_chains, ag_chains = parse_partner_groups(str(row.get("partners", "")))
    node_features = []
    node_positions = []
    solvent_props = []
    engineered_vec = extract_engineered_features(row)
    for idx in node_indices:
        residue = structure.residues[idx]
        one_hot = [
            1.0 if residue.one_letter == aa else 0.0
            for aa in STANDARD_AMINO_ACIDS
        ]
        is_antibody = 1.0 if residue.chain_id in ab_chains else 0.0
        partner_chains = (
            ag_chains
            if residue.chain_id in ab_chains
            else ab_chains
            if residue.chain_id in ag_chains
            else ag_chains or ab_chains
        )
        contact_count = 0
        dist_to_partner = float("inf")
        if partner_chains:
            partner_indices = structure.indices_for_chain_set(partner_chains)
            if partner_indices:
                dists = structure.distances_to_point(residue.centroid)
                partner_dists = dists[partner_indices]
                contact_count = int(np.sum(partner_dists <= interface_threshold))
                if partner_dists.size:
                    dist_to_partner = float(np.min(partner_dists))

        interface_flag = 1.0 if contact_count > 0 else 0.0
        mutated_flag = 1.0 if (residue.chain_id, residue.resseq) in mutated_keys else 0.0
        dist_center = float(np.linalg.norm(residue.centroid - center_point))
        coords = residue.centroid

        neighbor_solvent = structure.count_neighbors(residue.centroid, SOLVENT_RADIUS)
        solvent_proxy = 1.0 / (1.0 + neighbor_solvent)
        heavy_atoms = float(
            sum(1 for elem, _ in residue.atoms if elem.upper() != "H" and elem != "")
        )
        b_factors = [bf for _, bf in residue.atoms]
        avg_bfactor = float(np.mean(b_factors)) if b_factors else 0.0

        feature_vector = np.concatenate(
            [
                np.array(one_hot, dtype=float),
                np.array(
                    [
                        is_antibody,
                        interface_flag,
                        mutated_flag,
                        float(contact_count),
                        dist_center,
                        dist_to_partner if np.isfinite(dist_to_partner) else 0.0,
                    ],
                    dtype=float,
                ),
                np.array(
                    [solvent_proxy, heavy_atoms, avg_bfactor],
                    dtype=float,
                ),
                coords.astype(float),
                engineered_vec,
            ]
        )
        solvent_props.append(solvent_proxy)
        node_features.append(feature_vector)
        node_positions.append(coords)

    node_features = np.stack(node_features)
    node_positions = np.stack(node_positions)
    n_nodes = node_features.shape[0]
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    edge_attr: list[list[float]] = []
    for i in range(n_nodes):
        edges.append((i, i))
        weights.append(1.0)
        edge_attr.append([0.0, 0.0, 0.0, 0.0])
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = float(np.linalg.norm(node_positions[i] - node_positions[j]))
            if dist <= edge_radius:
                weight = float(np.exp(-dist / 5.0))
                edges.append((i, j))
                weights.append(weight)
                edges.append((j, i))
                weights.append(weight)
                solvent_diff = abs(solvent_props[i] - solvent_props[j])
                chain_flag = (
                    1.0
                    if structure.residues[node_indices[i]].chain_id
                    != structure.residues[node_indices[j]].chain_id
                    else 0.0
                )
                interface_flag = 1.0 if dist <= interface_threshold else 0.0
                attr = [dist, interface_flag, chain_flag, solvent_diff]
                edge_attr.append(attr)
                edge_attr.append(attr)

    edge_index = np.array(edges, dtype=int).T
    edge_weight = np.array(weights, dtype=float)
    edge_attr_arr = np.array(edge_attr, dtype=float)

    return {
        "sample_id": int(row["sample_id"]),
        "pdb_id": pdb_id,
        "split": str(row["split"]),
        "y": float(row["ddg"]),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_attr": edge_attr_arr,
    }


def build_graphs(
    input_path: Path,
    output_path: Path,
    graph_radius: float,
    edge_radius: float,
    interface_threshold: float,
) -> None:
    df = pd.read_csv(input_path)
    cache = StructureCache()
    graphs = []
    for _, row in df.iterrows():
        graph = build_graph(
            row,
            cache,
            graph_radius,
            edge_radius,
            interface_threshold,
        )
        if graph is not None:
            graphs.append(graph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(graphs, handle)
    print(f"[INFO] Saved {len(graphs)} graphs to {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build interface graphs from AB-Bind structures."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_PATH,
        help="CSV file with engineered AB-Bind features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Where to store serialized graphs.",
    )
    parser.add_argument(
        "--graph-radius",
        type=float,
        default=GRAPH_RADIUS,
        help="Radius around mutation center to retain residues.",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=EDGE_RADIUS,
        help="Cutoff for connecting residues via edges.",
    )
    parser.add_argument(
        "--interface-threshold",
        type=float,
        default=INTERFACE_THRESHOLD,
        help="Distance to consider a residue as part of the interface.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    build_graphs(
        args.features,
        args.output,
        args.graph_radius,
        args.edge_radius,
        args.interface_threshold,
    )


if __name__ == "__main__":
    main()
