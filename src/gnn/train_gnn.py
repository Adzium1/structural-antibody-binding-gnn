#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.structure_utils import SUBPROJECT_ROOT

GRAPH_DATA_PATH = SUBPROJECT_ROOT / "data" / "graphs" / "ab_bind_graphs.pkl"
METRICS_PATH = SUBPROJECT_ROOT / "reports" / "gnn_metrics.csv"


class GraphSample(Sequence):
    def __init__(self, data: dict):
        self.x = torch.as_tensor(data["node_features"], dtype=torch.float32)
        self.edge_index = torch.as_tensor(data["edge_index"], dtype=torch.long)
        self.edge_weight = torch.as_tensor(data["edge_weight"], dtype=torch.float32)
        self.edge_attr = torch.as_tensor(data["edge_attr"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        self.split = data["split"]

    def __getitem__(self, idx):
        return (self.x[idx], self.edge_index[idx], self.edge_weight[idx])

    def __len__(self):
        return int(self.x.shape[0])


class GraphCollection:
    def __init__(self, path: Path):
        with open(path, "rb") as handle:
            raw_graphs = pickle.load(handle)
        self.graphs = [GraphSample(graph) for graph in raw_graphs]
        self.split_index = {"train": [], "val": [], "test": []}
        for graph in self.graphs:
            if graph.split in self.split_index:
                self.split_index[graph.split].append(graph)

    def get_split(self, split: str) -> list[GraphSample]:
        return self.split_index.get(split, [])


class InterfaceGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_attr_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att_src = nn.Linear(hidden_dim, hidden_dim)
        self.att_dst = nn.Linear(hidden_dim, hidden_dim)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.res_lin = nn.Linear(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        self.layers = layers

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for _ in range(self.layers):
            h = F.relu(h)
            src = h[edge_index[0]]
            dst = h[edge_index[1]]
            bias = self.edge_mlp(edge_attr)
            att = torch.sigmoid(
                (self.att_src(src) + self.att_dst(dst) + bias).sum(dim=1, keepdim=True)
            )
            weight = edge_weight.unsqueeze(-1)
            msg = self.msg_lin(src) * att * weight
            agg = torch.zeros_like(h)
            agg = agg.index_add(0, edge_index[1], msg)
            h = F.relu(self.res_lin(h + agg))
        graph_rep = h.mean(dim=0)
        return self.readout(graph_rep).squeeze(-1)


def train_epoch(model: nn.Module, graphs: Sequence[GraphSample], optimizer: torch.optim.Optimizer) -> float:
    model.train()
    losses: list[float] = []
    criterion = nn.MSELoss()
    for graph in graphs:
        optimizer.zero_grad()
        pred = model(
            graph.x, graph.edge_index, graph.edge_weight, graph.edge_attr
        )
        loss = criterion(pred, graph.y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model: nn.Module, graphs: Sequence[GraphSample]) -> dict[str, float]:
    model.eval()
    truths: list[float] = []
    preds: list[float] = []
    with torch.no_grad():
        for graph in graphs:
            output = model(
                graph.x, graph.edge_index, graph.edge_weight, graph.edge_attr
            )
            preds.append(float(output.item()))
            truths.append(float(graph.y.item()))
    if not truths:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    mae = mean_absolute_error(truths, preds)
    rmse = mean_squared_error(truths, preds, squared=False)
    r2 = r2_score(truths, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple GNN on AB-Bind interface graphs.")
    parser.add_argument(
        "--graphs",
        type=Path,
        default=GRAPH_DATA_PATH,
        help="Pickle file produced by src.data.build_interface_graphs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for the message-passing layers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=METRICS_PATH,
        help="CSV file to log train/val/test metrics.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (number of epochs without val R² improvement).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = GraphCollection(args.graphs)
    sample_graphs = dataset.graphs
    if not sample_graphs:
        raise ValueError("No graphs were loaded. Run src.data.build_interface_graphs first.")

    input_dim = sample_graphs[0].x.shape[1]
    edge_attr_dim = sample_graphs[0].edge_attr.shape[1]
    model = InterfaceGNN(
        input_dim=input_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_records: list[dict[str, object]] = []
    best_val_r2 = float("-inf")
    best_epoch = 0
    best_summary: dict[str, object] | None = None
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, dataset.get_split("train"), optimizer)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss}
        for split in ("train", "val", "test"):
            metrics = evaluate(model, dataset.get_split(split))
            epoch_metrics.update(
                {
                    f"{split}_mae": metrics["mae"],
                    f"{split}_rmse": metrics["rmse"],
                    f"{split}_r2": metrics["r2"],
                }
            )
        val_r2 = epoch_metrics["val_r2"]
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            best_summary = epoch_metrics.copy()
            no_improve = 0
            epoch_metrics["is_best_epoch"] = True
        else:
            no_improve += 1
            epoch_metrics["is_best_epoch"] = False

        print(
            f"[epoch {epoch}] loss={train_loss:.3f} val_mae={epoch_metrics['val_mae']:.3f} "
            f"val_r2={val_r2:.3f}"
        )

        epoch_records.append(epoch_metrics)

        if no_improve >= args.patience:
            print(f"[INFO] Early stopping after epoch {epoch} (patience={args.patience}).")
            break

    if best_summary is not None:
        print(
            "[INFO] Best epoch %d (val_r2=%.3f, test_mae=%.3f, test_r2=%.3f)"
            % (best_epoch, best_summary["val_r2"], best_summary["test_mae"], best_summary["test_r2"])
        )

    metrics_df = pd.DataFrame(epoch_records)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)
    print(f"[INFO] Saved GNN metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
