"""
Phase 2 Spatio-Temporal Graph Attention Network (ST-GAT) Training & Evaluation Harness.
Trains ST-GAT with masked MAE loss, validation early stopping, and benchmarks
performance against Phase 1 classical baselines (Persistence, Ridge, XGBoost).
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (
    PILOT_SENSORS,
    MODELS_DIR,
    RANDOM_SEED,
    EPSILON_MAPE,
)
from src.data.loader import load_metr_la_raw, load_sensor_graph
from src.models_phase2.st_gat_model import SpatioTemporalGAT
from src.models_phase2.graph_dataset import create_dataloaders


def masked_mae_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes MAE exclusively over valid observations (mask == True).
    """
    diff = torch.abs(y_pred - y_true) * mask.float()
    return torch.sum(diff) / (torch.sum(mask.float()) + eps)


def masked_rmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes RMSE exclusively over valid observations.
    """
    sq_diff = torch.square(y_pred - y_true) * mask.float()
    mse = torch.sum(sq_diff) / (torch.sum(mask.float()) + eps)
    return torch.sqrt(mse)


def masked_mape(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps_val: float = EPSILON_MAPE,
) -> torch.Tensor:
    """
    Computes masked MAPE with epsilon clamping to avoid division by near-zero speeds.
    """
    clamped_true = torch.clamp(y_true, min=eps_val)
    ratio = (torch.abs(y_pred - y_true) / clamped_true) * mask.float()
    return (torch.sum(ratio) / (torch.sum(mask.float()) + 1e-6)) * 100.0


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    adj_tensor: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates ST-GAT across all horizons (15m, 30m, 60m).
    """
    model.eval()
    horizon_names = ["15min", "30min", "60min"]
    all_preds = []
    all_trues = []
    all_masks = []

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            preds, _ = model(x, adj_tensor)
            all_preds.append(preds.cpu())
            all_trues.append(y.cpu())
            all_masks.append(mask.cpu())

    cat_preds = torch.cat(all_preds, dim=0)
    cat_trues = torch.cat(all_trues, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)

    results = {}
    for h_idx, h_name in enumerate(horizon_names):
        p_h = cat_preds[:, :, h_idx]
        t_h = cat_trues[:, :, h_idx]
        m_h = cat_masks[:, :, h_idx]

        mae = float(masked_mae_loss(p_h, t_h, m_h).item())
        rmse = float(masked_rmse(p_h, t_h, m_h).item())
        mape = float(masked_mape(p_h, t_h, m_h).item())

        results[h_name] = {"mae": mae, "rmse": rmse, "mape": mape}

    return results


def train_st_gat(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    use_pilot_only: bool = True,
):
    """
    Main training execution function.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f">>> Phase 2: Training Spatio-Temporal GAT on {device}")
    print(f"=======================================================")

    # 1. Load Data
    print("Loading raw METR-LA speed matrix and sensor graph...")
    df = load_metr_la_raw()
    graph_meta = load_sensor_graph()

    sensor_ids = PILOT_SENSORS if use_pilot_only else graph_meta["sensor_ids"]
    full_adj = graph_meta["adj_mx"]
    all_sensor_ids = graph_meta["sensor_ids"]
    id_to_idx = {s: i for i, s in enumerate(all_sensor_ids)}

    # Subgraph adjacency matrix for pilot sensors
    indices = [id_to_idx[s] for s in sensor_ids]
    adj_sub = full_adj[np.ix_(indices, indices)]
    # Add self-loops to adjacency
    np.fill_diagonal(adj_sub, 1.0)

    # 2. Build DataLoaders
    print("Creating chronological train/val/test graph DataLoaders...")
    loaders, adj_tensor, (train_mean, train_std) = create_dataloaders(
        df=df,
        sensor_ids=sensor_ids,
        adj_mx=adj_sub,
        batch_size=batch_size,
    )
    adj_tensor = adj_tensor.to(device)

    # 3. Instantiate Model
    num_nodes = len(sensor_ids)
    model = SpatioTemporalGAT(
        num_nodes=num_nodes,
        in_features=5,
        temporal_hidden_dim=64,
        spatial_hidden_dim=32,
        num_heads=4,
        num_horizons=3,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_mae = float("inf")
    patience_counter = 0
    save_path = MODELS_DIR / "st_gat_best_checkpoint.pt"

    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} trainable parameters.")
    print(f"Beginning training loop for {epochs} epochs (Early stopping patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        t0 = time.time()

        for x_batch, y_batch, mask_batch in loaders["train"]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            preds, _ = model(x_batch, adj_tensor)
            loss = masked_mae_loss(preds, y_batch, mask_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)

        # Validation evaluation
        val_metrics = evaluate_model(model, loaders["val"], adj_tensor, device)
        avg_val_mae = (val_metrics["15min"]["mae"] + val_metrics["30min"]["mae"] + val_metrics["60min"]["mae"]) / 3.0

        scheduler.step(avg_val_mae)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train MAE: {avg_train_loss:.4f} | "
            f"Val MAE (15m: {val_metrics['15min']['mae']:.2f}, 30m: {val_metrics['30min']['mae']:.2f}, 60m: {val_metrics['60min']['mae']:.2f}) | "
            f"Avg Val: {avg_val_mae:.4f}"
        )

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "sensor_ids": sensor_ids,
                },
                save_path,
            )
            print(f"  --> Saved new best checkpoint to {save_path.name} (Val MAE: {best_val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (Patience={patience}).")
                break

    # 4. Final Test Set Evaluation
    print("\n=======================================================")
    print(">>> Final Held-Out Test Evaluation")
    print("=======================================================")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(model, loaders["test"], adj_tensor, device)

    print("\nPhase 2 ST-GAT Test Results vs. Phase 1 Classical Benchmarks:")
    print("-" * 75)
    print(f"{'Model':<22} | {'Horizon':<8} | {'MAE (mph)':<10} | {'RMSE (mph)':<10} | {'MAPE (%)':<8}")
    print("-" * 75)
    print(f"{'Phase 1: Persistence':<22} | 15 min   | 3.0011     | 5.5959     | 7.12%")
    print(f"{'Phase 1: Ridge':<22} | 15 min   | 2.8789     | 5.3466     | 7.45%")
    print(f"{'Phase 1: XGBoost':<22} | 15 min   | 2.5909     | 4.8718     | 6.68%")
    print(f"{'Phase 2: ST-GAT (Ours)':<22} | 15 min   | {test_metrics['15min']['mae']:<10.4f} | {test_metrics['15min']['rmse']:<10.4f} | {test_metrics['15min']['mape']:.2f}%")
    print("-" * 75)
    print(f"{'Phase 1: Persistence':<22} | 30 min   | 3.6157     | 7.2034     | 9.04%")
    print(f"{'Phase 1: Ridge':<22} | 30 min   | 3.6050     | 6.8440     | 10.12%")
    print(f"{'Phase 1: XGBoost':<22} | 30 min   | 3.1668     | 6.0589     | 8.69%")
    print(f"{'Phase 2: ST-GAT (Ours)':<22} | 30 min   | {test_metrics['30min']['mae']:<10.4f} | {test_metrics['30min']['rmse']:<10.4f} | {test_metrics['30min']['mape']:.2f}%")
    print("-" * 75)
    print(f"{'Phase 1: Persistence':<22} | 60 min   | 4.7302     | 9.6660     | 12.68%")
    print(f"{'Phase 1: Ridge':<22} | 60 min   | 4.9066     | 8.9476     | 15.02%")
    print(f"{'Phase 1: XGBoost':<22} | 60 min   | 4.0083     | 7.5269     | 11.64%")
    print(f"{'Phase 2: ST-GAT (Ours)':<22} | 60 min   | {test_metrics['60min']['mae']:<10.4f} | {test_metrics['60min']['rmse']:<10.4f} | {test_metrics['60min']['mape']:.2f}%")
    print("-" * 75)

    return test_metrics


if __name__ == "__main__":
    train_st_gat(epochs=10, batch_size=32)

