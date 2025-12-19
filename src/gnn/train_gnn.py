#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

from src.data.structure_utils import SUBPROJECT_ROOT

GRAPH_DATA_PATH = SUBPROJECT_ROOT / "data" / "graphs" / "ab_bind_graphs.pkl"
METRICS_PATH = SUBPROJECT_ROOT / "reports" / "gnn_metrics.csv"
SCALER_PATH = SUBPROJECT_ROOT / "reports" / "ddg_scaler.pkl"

EDGE_ATTR_BACKFILL_WARNED = False


class GraphSample(Sequence):
    def __init__(self, data: dict):
        self.x = torch.as_tensor(data["node_features"], dtype=torch.float32)
        self.edge_index = torch.as_tensor(data["edge_index"], dtype=torch.long)
        self.edge_weight = torch.as_tensor(data["edge_weight"], dtype=torch.float32)

        # Older graph pickles may lack edge_attr; backfill with zeros to allow training,
        # but warn so users can regenerate graphs with src.data.build_interface_graphs.
        global EDGE_ATTR_BACKFILL_WARNED
        if "edge_attr" in data:
            self.edge_attr = torch.as_tensor(data["edge_attr"], dtype=torch.float32)
        else:
            num_edges = int(self.edge_index.shape[1])
            self.edge_attr = torch.zeros((num_edges, 1), dtype=torch.float32)
            if not EDGE_ATTR_BACKFILL_WARNED:
                print(
                    "[WARN] edge_attr missing in graph data; using zeros. "
                    "Regenerate graphs via `python -m src.data.build_interface_graphs` "
                    "for full edge features."
                )
                EDGE_ATTR_BACKFILL_WARNED = True

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
        hidden_dim: int = 256,
        layers: int = 6,
        dropout: float = 0.2,
        use_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = layers
        self.dropout = nn.Dropout(p=dropout)

        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(edge_attr_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.att_src = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.att_dst = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.msg_lin = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.res_lin = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        if use_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.readout = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming initialization for stability with ReLU activations.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.readout:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp edge attributes and weights to control activation scale
        edge_attr = torch.tanh(edge_attr)
        edge_weight = torch.clamp(edge_weight, min=0.0, max=1.0)

        h = self.input_proj(x)
        for idx in range(self.layers):
            h_act = F.relu(h)
            src = h_act[edge_index[0]]
            dst = h_act[edge_index[1]]
            bias = self.edge_mlps[idx](edge_attr)
            att = torch.sigmoid(
                (self.att_src[idx](src) + self.att_dst[idx](dst) + bias).sum(dim=1, keepdim=True)
            )
            weight = edge_weight.unsqueeze(-1)
            msg = self.msg_lin[idx](src) * att * weight
            agg = torch.zeros_like(h_act)
            agg = agg.index_add(0, edge_index[1], msg)
            updated = self.res_lin[idx](h_act + agg)
            updated = self.dropout(F.relu(updated))
            h = self.norms[idx](updated)
        graph_rep = h.mean(dim=0)
        # Scale down the final output to avoid huge initial predictions
        return (self.readout(graph_rep) / 50.0).squeeze(-1)


def collect_grad_stats(model: nn.Module) -> dict[str, dict[str, float]]:
    """
    Capture basic gradient statistics for each parameter tensor.
    """
    stats: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        stats[name] = {
            "mean": float(grad.abs().mean().item()),
            "max": float(grad.abs().max().item()),
            "norm": float(grad.norm().item()),
        }
    return stats


def train_epoch(
    model: nn.Module,
    graphs: Sequence[GraphSample],
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    track_grads: bool = False,
) -> tuple[float, dict[str, dict[str, float]]]:
    model.train()
    losses: list[float] = []
    grad_stats: dict[str, dict[str, float]] = {}
    for graph in graphs:
        optimizer.zero_grad()
        pred = model(
            graph.x, graph.edge_index, graph.edge_weight, graph.edge_attr
        )
        pred_vec = pred.view(-1)
        target_vec = graph.y.view(-1)
        loss = criterion(pred_vec, target_vec)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if track_grads:
            grad_stats = collect_grad_stats(model)
        optimizer.step()
        losses.append(float(loss.item()))
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, grad_stats


def evaluate(
    model: nn.Module,
    graphs: Sequence[GraphSample],
    scaler: StandardScaler | None = None,
) -> dict[str, float]:
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
    preds_arr = np.array(preds)
    truths_arr = np.array(truths)
    if scaler is not None:
        preds_arr = scaler.inverse_transform(preds_arr.reshape(-1, 1)).flatten()
        truths_arr = scaler.inverse_transform(truths_arr.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(truths_arr, preds_arr)
    rmse = mean_squared_error(truths_arr, preds_arr, squared=False)
    r2 = r2_score(truths_arr, preds_arr)
    return {"mae": mae, "rmse": rmse, "r2": r2}


class DistributionWeightedMSE(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = float(mean)
        self.std = float(max(std, 1e-6))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(-((target - self.mean) ** 2) / (2 * (self.std ** 2)))
        return torch.mean(weights * (pred - target) ** 2)


def build_loss_fn(train_targets: np.ndarray, loss_type: str) -> nn.Module:
    if loss_type == "dist_weighted":
        return DistributionWeightedMSE(train_targets.mean(), train_targets.std())
    return nn.MSELoss()


def standardize_dataset_targets(dataset: GraphCollection) -> StandardScaler:
    train_graphs = dataset.get_split("train")
    if not train_graphs:
        raise ValueError("No training graphs available for target standardization.")
    train_targets = torch.stack([g.y for g in train_graphs]).numpy().reshape(-1, 1)
    scaler = StandardScaler().fit(train_targets)
    for graph in dataset.graphs:
        scaled = scaler.transform(graph.y.numpy().reshape(-1, 1)).flatten()
        graph.y = torch.tensor(scaled, dtype=torch.float32)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Saved target scaler to {SCALER_PATH}")
    return scaler


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
        default=80,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for the message-passing layers.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="Number of message-passing layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied after each layer.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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
        default=20,
        help="Early stopping patience (number of epochs without val R² improvement).",
    )
    parser.add_argument(
        "--standardize-targets",
        action="store_true",
        default=True,
        help="Standardize ddG targets using train split statistics.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["mse", "dist_weighted"],
        default="mse",
        help="Loss function variant.",
    )
    return parser.parse_args()


def plot_training_curves(
    records: list[dict[str, object]],
    grad_logs: list[dict[str, dict[str, float]]],
    plot_path: Path,
) -> None:
    """
    Save simple training curves (train loss + val R2) and gradient norms if available.
    """
    if not records:
        return
    epochs = [rec["epoch"] for rec in records]
    train_loss = [rec["train_loss"] for rec in records]
    val_r2 = [rec["val_r2"] for rec in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Train loss")

    axes[1].plot(epochs, val_r2, label="val_r2", color="green")
    axes[1].axhline(0.0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("R2")
    axes[1].set_title("Validation R2")

    # Optional gradient norm trace for the first parameter
    if grad_logs:
        first_epoch = next((g for g in grad_logs if g), None)
        if first_epoch:
            first_param = next(iter(first_epoch.keys()))
            grad_norms = [
                g.get(first_param, {}).get("norm", float("nan")) if g else float("nan")
                for g in grad_logs
            ]
            axes[0].twinx().plot(
                epochs,
                grad_norms,
                label="grad_norm",
                color="orange",
                alpha=0.4,
            )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def overfit_debug_run(
    dataset: GraphCollection,
    input_dim: int,
    edge_attr_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    n_samples: int,
    epochs: int = 100,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    use_norm: bool = False,
) -> float:
    """
    Small-capacity test: can the model memorize a tiny subset?
    Returns final training loss.
    """
    subset = dataset.get_split("train")[:n_samples]
    if not subset:
        print("[WARN] No training graphs available for overfit debug.")
        return float("nan")
    model = InterfaceGNN(
        input_dim=input_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        use_norm=use_norm,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for graph in subset:
            optimizer.zero_grad()
            pred = model(
                graph.x, graph.edge_index, graph.edge_weight, graph.edge_attr
            )
            pred_vec = pred.view(-1)
            target_vec = graph.y.view(-1)
            loss = criterion(pred_vec, target_vec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch % 20 == 0:
            print(f"[debug overfit] epoch {epoch} loss={np.mean(losses):.6f}")
        if np.mean(losses) < 1e-4:
            print(f"[debug overfit] memorized subset at epoch {epoch}")
            break
    final_loss = float(np.mean(losses))
    print(f"[debug overfit] final loss={final_loss:.6f}")
    return final_loss


def ultra_minimal_probe(
    dataset: GraphCollection,
    input_dim: int,
    edge_attr_dim: int,
    hidden_dim: int = 64,
    layers: int = 2,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> float:
    """
    Fast probe on a single smallest graph to ensure the training loop is functional.
    """
    if not dataset.graphs:
        raise ValueError("Dataset is empty.")
    graph = dataset.graphs[0]
    model = InterfaceGNN(
        input_dim=input_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=0.0,
        use_norm=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(graph.x, graph.edge_index, graph.edge_weight, graph.edge_attr).view(-1)
        target = graph.y.view(-1)
        loss = F.mse_loss(pred, target)
        if torch.isnan(loss):
            print(f"[ultra probe] NaN at epoch {epoch}")
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        if epoch % max(1, epochs // 5) == 0:
            print(f"[ultra probe] epoch {epoch} loss={loss.item():.6f}")
    return float(loss.item())


def main() -> None:
    args = parse_arguments()
    dataset = GraphCollection(args.graphs)
    sample_graphs = dataset.graphs
    if not sample_graphs:
        raise ValueError("No graphs were loaded. Run src.data.build_interface_graphs first.")

    scaler: StandardScaler | None = None
    if args.standardize_targets:
        scaler = standardize_dataset_targets(dataset)

    input_dim = sample_graphs[0].x.shape[1]
    edge_attr_dim = sample_graphs[0].edge_attr.shape[1]
    model = InterfaceGNN(
        input_dim=input_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_targets_arr = np.array(
        [float(g.y.item()) for g in dataset.get_split("train")]
    )
    loss_fn = build_loss_fn(train_targets_arr, args.loss_type)

    epoch_records: list[dict[str, object]] = []
    grad_logs: list[dict[str, dict[str, float]]] = []
    best_val_r2 = float("-inf")
    best_epoch = 0
    best_summary: dict[str, object] | None = None
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, grad_stats = train_epoch(
            model,
            dataset.get_split("train"),
            optimizer,
            loss_fn,
            track_grads=True,
        )
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss}
        grad_logs.append(grad_stats)
        for split in ("train", "val", "test"):
            metrics = evaluate(model, dataset.get_split(split), scaler=scaler)
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

    diag_path = SUBPROJECT_ROOT / "reports" / "gnn_diagnostics.json"
    plot_path = SUBPROJECT_ROOT / "reports" / "gnn_training_curves.png"
    diagnostics = {
        "epoch_records": epoch_records,
        "gradient_logs": grad_logs,
        "best_epoch": best_summary,
    }
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diagnostics, indent=2))
    print(f"[INFO] Saved diagnostics to {diag_path}")
    plot_training_curves(epoch_records, grad_logs, plot_path)
    print(f"[INFO] Saved training curves to {plot_path}")

    # Optional memorization test to check capacity/optimization
    overfit_loss = overfit_debug_run(
        dataset,
        input_dim=input_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        n_samples=min(10, len(dataset.get_split("train"))),
        epochs=200,
        lr=max(args.lr, 5e-3),
    )
    if overfit_loss > 1e-3:
        print("[WARN] Overfit debug did not reach near-zero loss; consider increasing capacity or lr.")


if __name__ == "__main__":
    main()
