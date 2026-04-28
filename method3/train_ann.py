"""
=============================================================================
Method 3: ANN Baseline (PyTorch / Standard CNN)
=============================================================================
Task:       Image classification on MNIST (10 classes)
Model:      Convolutional ANN with ReLU activations
            Architecture: 12C5 - MP2 - 32C5 - MP2 - FC800 - FC10
            (IDENTICAL depth/width to Methods 1 & 2, but ReLU instead of LIF)
Framework:  Plain PyTorch (no SNN libraries)

PURPOSE:
    This serves as the performance CEILING and timing REFERENCE:
    - Accuracy ceiling: How well can this architecture do without spikes?
    - Timing floor: How fast is training without the T-timestep overhead?
    Comparing Methods 1 & 2 against this baseline quantifies the "cost of
    going spiking" in both accuracy and wall-clock time.

Usage:
    python train_ann.py --device cuda:0
    python train_ann.py --device cpu

Outputs:
    results/ann_timing_run{R}.csv   - per-epoch timing data
    results/ann_summary.csv         - summary across all runs
=============================================================================
"""

import os
import time
import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Method 3: ANN Baseline MNIST Training")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to train on (cuda:0 or cpu)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download/store MNIST")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory to save timing results")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of repeated runs for variance")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET (identical to Methods 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size):
    """
    Load MNIST with standard normalization.
    IDENTICAL preprocessing to Methods 1 & 2 for fair comparison.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=2,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, drop_last=True, num_workers=2,
                             pin_memory=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class ConvANN(nn.Module):
    """
    Standard convolutional ANN for MNIST classification.

    Architecture (matched to Methods 1 & 2 in depth and parameter count):
        Conv2d(1, 12, 5)  → ReLU → MaxPool(2)
        Conv2d(12, 32, 5) → ReLU → MaxPool(2)
        Flatten → Linear(32*4*4, 800) → ReLU
        Linear(800, 10)

    KEY DIFFERENCES from SNN methods:
        - ReLU replaces LIF neurons (continuous activations, no spikes)
        - NO timestep loop (single forward pass per image)
        - Standard backpropagation (no BPTT needed)
        - This is the standard deep learning approach

    Communication pattern:
        - Pure feed-forward, no temporal dependency
        - Single CUDA kernel sequence per forward pass
        - No sequential timestep overhead
        - Parallel across batch dimension only
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: Conv → ReLU → Pool
            nn.Conv2d(1, 12, 5),        # 28x28 → 24x24
            nn.ReLU(),
            nn.MaxPool2d(2),             # 24x24 → 12x12

            # Layer 2: Conv → ReLU → Pool
            nn.Conv2d(12, 32, 5),       # 12x12 → 8x8
            nn.ReLU(),
            nn.MaxPool2d(2),             # 8x8   → 4x4

            # Flatten + FC layers
            nn.Flatten(),

            # Layer 3: FC → ReLU
            nn.Linear(32 * 4 * 4, 800),
            nn.ReLU(),

            # Layer 4 (output): FC only (no activation — raw logits for CrossEntropy)
            nn.Linear(800, 10),
        )

    def forward(self, x):
        """
        Standard single-pass forward.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Logits of shape (batch, 10)

        NOTE: No timestep loop, no membrane state, no spike generation.
              This is the simplest possible forward pass for this architecture.
        """
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIMING UTILITIES (identical to Methods 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────
class GPUTimer:
    """Precise GPU timing using CUDA events. Falls back to wall-clock on CPU."""
    def __init__(self, device):
        self.use_cuda = device.type == "cuda"
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda:
            self.start_event.record()
        else:
            self._start = time.perf_counter()

    def stop(self):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        else:
            return time.perf_counter() - self._start


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """
    Train for one epoch. Returns timing breakdown and loss.
    No timestep loop — single forward + backward per batch.
    """
    model.train()
    timer = GPUTimer(device)

    epoch_loss = 0.0
    num_batches = 0
    total_forward = 0.0
    total_backward = 0.0
    epoch_start = time.perf_counter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # ── Forward pass timing ──
        timer.start()
        output = model(data)                  # Single forward pass
        loss = loss_fn(output, target)        # Single loss computation
        forward_time = timer.stop()
        total_forward += forward_time

        # ── Backward pass timing ──
        timer.start()
        optimizer.zero_grad()
        loss.backward()                       # Standard backprop (NOT BPTT)
        optimizer.step()
        backward_time = timer.stop()
        total_backward += backward_time

        epoch_loss += loss.item()
        num_batches += 1

    epoch_end = time.perf_counter()
    total_time = epoch_end - epoch_start
    avg_loss = epoch_loss / num_batches

    return {
        "forward_time": total_forward,
        "backward_time": total_backward,
        "total_time": total_time,
        "data_loading_time": total_time - total_forward - total_backward,
        "avg_loss": avg_loss,
        "num_batches": num_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, test_loader, device):
    """Evaluate accuracy on test set."""
    model.eval()
    correct = 0
    total = 0

    timer = GPUTimer(device)
    timer.start()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            predicted = output.argmax(dim=1)

            correct += (predicted == target).sum().item()
            total += target.size(0)

    inference_time = timer.stop()
    accuracy = 100.0 * correct / total

    return accuracy, inference_time


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"MNIST loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {len(train_loader)}")
    print(f"Runs: {args.num_runs}")
    print(f"NOTE: No timestep sweep — ANN has no T parameter")
    print("=" * 70)

    summary_rows = []

    for run in range(args.num_runs):
        print(f"\n{'─' * 70}")
        print(f"  Run {run + 1}/{args.num_runs}")
        print(f"{'─' * 70}")

        # Fresh model + optimizer
        model = ConvANN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if run == 0:
            print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        epoch_rows = []

        # GPU warmup
        if device.type == "cuda":
            warmup_data, _ = next(iter(train_loader))
            warmup_data = warmup_data.to(device)
            _ = model(warmup_data)
            torch.cuda.synchronize()

        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer,
                                          loss_fn, device)
            test_acc, inference_time = evaluate(model, test_loader, device)

            row = {
                "T": 1,  # ANN = effectively T=1 (single pass)
                "run": run + 1,
                "epoch": epoch,
                "forward_time_sec": round(train_stats["forward_time"], 4),
                "backward_time_sec": round(train_stats["backward_time"], 4),
                "data_loading_time_sec": round(train_stats["data_loading_time"], 4),
                "total_epoch_time_sec": round(train_stats["total_time"], 4),
                "train_loss": round(train_stats["avg_loss"], 6),
                "test_accuracy_pct": round(test_acc, 2),
                "inference_time_sec": round(inference_time, 4),
            }
            epoch_rows.append(row)

            print(f"    Epoch {epoch:2d}/{args.epochs} │ "
                  f"Loss: {train_stats['avg_loss']:.4f} │ "
                  f"Acc: {test_acc:.2f}% │ "
                  f"Train: {train_stats['total_time']:.2f}s "
                  f"(fwd: {train_stats['forward_time']:.2f}s, "
                  f"bwd: {train_stats['backward_time']:.2f}s)")

        # Save per-epoch CSV
        csv_path = os.path.join(args.results_dir, f"ann_timing_run{run + 1}.csv")
        save_csv(csv_path, epoch_rows)
        print(f"  Saved: {csv_path}")

        final = epoch_rows[-1]
        final["total_params"] = total_params
        summary_rows.append(final)

    # Save summary
    summary_path = os.path.join(args.results_dir, "ann_summary.csv")
    save_csv(summary_path, summary_rows)
    print(f"\n{'=' * 70}")
    print(f"Summary saved: {summary_path}")
    print(f"{'=' * 70}")

    # Print summary
    print(f"\n{'Run':>4} {'Epoch Time (s)':>15} {'Fwd (s)':>10} "
          f"{'Bwd (s)':>10} {'Accuracy':>10}")
    print("─" * 55)
    for row in summary_rows:
        print(f"{row['run']:>4} {row['total_epoch_time_sec']:>15.2f} "
              f"{row['forward_time_sec']:>10.2f} {row['backward_time_sec']:>10.2f} "
              f"{row['test_accuracy_pct']:>9.2f}%")


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
