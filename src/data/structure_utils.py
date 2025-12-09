from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser, is_aa

STANDARD_AMINO_ACIDS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

THREE_TO_ONE_MAP = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

MUTATION_PATTERN = re.compile(
    r"(?P<chain>[A-Za-z0-9]+):(?P<wild>[A-Z])(?P<pos>\d+)(?P<mut>[A-Z])"
)

SUBPROJECT_ROOT = Path(__file__).resolve().parents[2] / "3D-GNN-over-antibody-antigen"
PDB_DATABASE_DIR = SUBPROJECT_ROOT / "data" / "external" / "AB-Bind-Database"


@dataclass(frozen=True)
class ResidueDescriptor:
    chain_id: str
    resseq: int
    icode: str
    centroid: np.ndarray
    resname: str
    one_letter: str


class StructureInfo:
    def __init__(self, residues: Sequence[ResidueDescriptor]):
        self.residues = list(residues)
        if self.residues:
            self.centroids = np.stack([res.centroid for res in self.residues])
        else:
            self.centroids = np.zeros((0, 3), dtype=float)

        self.chain_map: dict[str, List[int]] = {}
        self.index_map: dict[Tuple[str, int], int] = {}
        for idx, residue in enumerate(self.residues):
            chain = residue.chain_id
            self.chain_map.setdefault(chain, []).append(idx)
            self.index_map[(chain, residue.resseq)] = idx

    def find_residue(self, chain_id: str, resseq: int) -> ResidueDescriptor | None:
        key = (chain_id, resseq)
        idx = self.index_map.get(key)
        if idx is None:
            return None
        return self.residues[idx]

    def indices_for_chain_set(self, chain_set: Iterable[str]) -> List[int]:
        indices = []
        for chain in chain_set:
            indices.extend(self.chain_map.get(chain, []))
        return indices

    def distances_to_point(self, point: np.ndarray) -> np.ndarray:
        if self.centroids.size == 0:
            return np.array([], dtype=float)
        return np.linalg.norm(self.centroids - point, axis=1)

    def count_neighbors(self, point: np.ndarray, radius: float) -> int:
        dists = self.distances_to_point(point)
        if dists.size == 0:
            return 0
        mask = dists <= radius
        zero_present = bool(dists.size and np.any(np.isclose(dists, 0.0)))
        return int(max(0, mask.sum() - (1 if zero_present else 0)))


class StructureCache:
    def __init__(self, pdb_dir: Path | None = None):
        self.pdb_dir = pdb_dir or PDB_DATABASE_DIR
        self._parser = PDBParser(PERMISSIVE=True, QUIET=True)
        self._cache: Mapping[str, StructureInfo] = {}

    def load(self, pdb_id: str) -> StructureInfo:
        pid = pdb_id.lower()
        if pid in self._cache:
            return self._cache[pid]
        pdb_path = self.pdb_dir / f"{pid}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found at {pdb_path}")
        try:
            structure = self._parser.get_structure(pid, str(pdb_path))
        except (TypeError, RuntimeError, ValueError) as exc:
            print(f"[WARN] Could not parse {pdb_path}: {exc}")
            info = StructureInfo([])
            self._cache[pid] = info
            return info
        residues: List[ResidueDescriptor] = []
        model = next(iter(structure))
        for chain in model:
            chain_id = (chain.id or "").strip() or " "
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                coords = np.array([atom.coord for atom in residue.get_atoms()])
                centroid = coords.mean(axis=0)
                resname = residue.get_resname()
                one_letter = THREE_TO_ONE_MAP.get(resname, "X")

                resseq = residue.get_id()[1]
                icode = residue.get_id()[2].strip()
                residues.append(
                    ResidueDescriptor(
                        chain_id=chain_id,
                        resseq=resseq,
                        icode=icode,
                        centroid=centroid,
                        resname=resname,
                        one_letter=one_letter,
                    )
                )

        info = StructureInfo(residues)
        self._cache[pid] = info
        return info


def parse_mutation_entry(entry: str) -> tuple[str, str, int, str] | None:
    if not entry:
        return None
    match = MUTATION_PATTERN.fullmatch(entry.strip())
    if not match:
        return None
    data = match.groupdict()
    pos = int(data["pos"])
    return data["chain"], data["wild"], pos, data["mut"]


def parse_mutation_list(mutation_text: str) -> list[tuple[str, str, int, str]]:
    if not mutation_text:
        return []
    tokens = [t.strip() for t in mutation_text.replace(";", ",").split(",") if t.strip()]
    parsed = []
    for token in tokens:
        entry = parse_mutation_entry(token)
        if entry is not None:
            parsed.append(entry)
    return parsed


def parse_partner_groups(partners_text: str) -> tuple[set[str], set[str]]:
    raw = (partners_text or "").replace(" ", "")
    if "_" not in raw:
        return set(raw), set()
    left, right = raw.split("_", 1)
    return set(left), set(right)
