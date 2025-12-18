#!/usr/bin/env python
"""
Lightweight debug harness for memorization stability:
- checks shape consistency,
- logs gradients per layer,
- detects NaN/Inf in activations,
- runs a short training loop on a tiny subset.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from src.gnn.train_gnn import (
    GraphCollection,
    InterfaceGNN,
    GRAPH_DATA_PATH,
)


def debug_memorization_run(
    graphs_path: Path,
    hidden_dim: int,
    layers: int,
    lr: float,
    steps: int = 10,
) -> None:
    dataset = GraphCollection(graphs_path)
    subset = dataset.get_split("train")[:10] or dataset.graphs[:10]
    if not subset:
        raise ValueError("No graphs available for debug.")
    g0 = subset[0]
    model = InterfaceGNN(
        input_dim=g0.x.shape[1],
        edge_attr_dim=g0.edge_attr.shape[1],
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # single "batch" = first graph for simplicity
    batch = subset[0]
    for step in range(steps):
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.edge_attr)
        pred_vec = pred.view(-1)
        target_vec = batch.y.view(-1)
        if pred_vec.shape != target_vec.shape:
            raise RuntimeError(
                f"Shape mismatch: pred {pred_vec.shape} vs target {target_vec.shape}"
            )
        loss = criterion(pred_vec, target_vec)
        print(f"[step {step}] loss={loss.item():.6f} pred_mean={pred_vec.mean().item():.3f}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARN] NaN/Inf loss detected; aborting.")
            break
        loss.backward()
        # log gradient norms
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gn = param.grad.data.norm(2).item()
            total_norm += gn * gn
            print(f"  grad {name}: {gn:.6f}")
        total_norm = total_norm ** 0.5
        print(f"  total grad norm: {total_norm:.6f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug memorization stability on tiny subset.")
    parser.add_argument("--graphs", type=Path, default=GRAPH_DATA_PATH)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    debug_memorization_run(
        args.graphs,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        lr=args.lr,
        steps=args.steps,
    )
