"""
=============================================================================
Method 1: Direct SNN Training via Surrogate Gradients (SNNTorch)
=============================================================================
Task:       Image classification on MNIST (10 classes)
Model:      Convolutional SNN with LIF neurons
            Architecture: 12C5 - MP2 - 32C5 - MP2 - FC800 - FC10
            (matches snnTorch tutorial / Spyx benchmark architecture)
Framework:  SNNTorch (PyTorch-based)
Surrogate:  Arctangent 

This script trains the SNN for multiple values of T (timesteps) and
records per-epoch timing breakdowns and accuracy for performance analysis.

Usage:
    python train_snntorch.py --device cuda:0
    python train_snntorch.py --device cpu    # fallback

Outputs:
    results/snntorch_timing_T{T}.csv   - per-epoch timing data
    results/snntorch_summary.csv       - summary across all T values
=============================================================================
"""

import os
import sys
import time
import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Method 1: SNNTorch MNIST Training")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to train on (cuda:0 or cpu)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per T value")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--beta", type=float, default=0.95,
                        help="LIF neuron decay rate (membrane potential leak)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download/store MNIST")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory to save timing results")
    parser.add_argument("--T_values", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64],
                        help="List of timestep values to sweep")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of repeated runs per T for variance")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size):
    """
    Load MNIST with standard normalization.
    Images are 28x28 grayscale, normalized to mean=0, std=1.
    No data augmentation — clean benchmark.
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
class SpikingCNN(nn.Module):
    """
    Convolutional SNN for MNIST classification.

    Architecture (matches standard SNN benchmark):
        Conv2d(1, 12, 5)  → LIF → MaxPool(2)
        Conv2d(12, 32, 5) → LIF → MaxPool(2)
        Flatten → Linear(32*4*4, 800) → LIF
        Linear(800, 10) → LIF (output)

    The forward pass loops over T timesteps. At each timestep, the SAME
    input image is fed into the network (rate coding via repeated
    presentation). This is the standard approach in SNNTorch.

    Communication pattern:
        - Sequential across T timesteps (each depends on prior membrane state)
        - Parallel across batch dimension (independent samples)
        - Within each timestep: standard CNN forward pass (parallel conv ops)
    """

    def __init__(self, beta=0.95):
        super().__init__()

        # Synaptic layers (weights — same as ANN)
        self.conv1 = nn.Conv2d(1, 12, 5)    # 28x28 → 24x24
        self.pool1 = nn.MaxPool2d(2)         # 24x24 → 12x12
        self.conv2 = nn.Conv2d(12, 32, 5)   # 12x12 → 8x8
        self.pool2 = nn.MaxPool2d(2)         # 8x8   → 4x4
        self.fc1 = nn.Linear(32 * 4 * 4, 800)
        self.fc2 = nn.Linear(800, 10)

        # Spiking neuron layers (LIF with arctangent surrogate gradient)
        spike_grad = surrogate.atan()
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, T):
        """
        Forward pass over T timesteps.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) — single static image
            T: Number of simulation timesteps

        Returns:
            spike_record: Output spikes at each timestep, shape (T, batch, 10)
            mem_record:   Output membrane potentials, shape (T, batch, 10)

        NOTE: This loop is the KEY parallel computing bottleneck.
              Each iteration launches separate GPU kernels for conv/linear ops,
              plus the LIF state update. The loop runs in Python, so there is
              Python-level overhead + GPU kernel launch overhead × T.
        """
        # Initialize membrane potentials to zero
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spike_record = []
        mem_record = []

        # ── Timestep loop (sequential — this is what SpikingJelly optimizes) ──
        for step in range(T):
            # Layer 1: Conv → LIF → Pool
            cur1 = self.pool1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2: Conv → LIF → Pool
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3: FC → LIF
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            # Layer 4 (output): FC → LIF
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spike_record.append(spk4)
            mem_record.append(mem4)

        return torch.stack(spike_record), torch.stack(mem_record)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIMING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
class GPUTimer:
    """
    Precise GPU timing using CUDA events.
    Falls back to wall-clock time on CPU.
    """
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
            return self.start_event.elapsed_time(self.end_event) / 1000.0  # ms → sec
        else:
            return time.perf_counter() - self._start


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, T):
    """
    Train for one epoch. Returns timing breakdown and loss.

    Timing breakdown:
        - forward_time:  Time for forward pass (T timestep loop + loss computation)
        - backward_time: Time for backward pass (BPTT across T timesteps)
        - total_time:    Wall-clock time for entire epoch (includes data loading)
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
        spike_record, mem_record = model(data, T)

        # Loss: cross-entropy on summed output membrane potential
        # (sum over T timesteps — standard SNNTorch approach)
        loss = torch.zeros(1, device=device)
        for step in range(T):
            loss += loss_fn(mem_record[step], target)
        forward_time = timer.stop()
        total_forward += forward_time

        # ── Backward pass timing ──
        timer.start()
        optimizer.zero_grad()
        loss.backward()
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
def evaluate(model, test_loader, device, T):
    """
    Evaluate accuracy on test set.
    Classification is based on the neuron with the highest total spike count
    across all T timesteps.
    """
    model.eval()
    correct = 0
    total = 0

    timer = GPUTimer(device)
    timer.start()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            spike_record, _ = model(data, T)

            # Sum spikes across all timesteps → shape (batch, 10)
            total_spikes = spike_record.sum(dim=0)
            predicted = total_spikes.argmax(dim=1)

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

    # Load data once (reused across all T values)
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"MNIST loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {len(train_loader)}")
    print(f"T values to sweep: {args.T_values}")
    print(f"Runs per T: {args.num_runs}")
    print("=" * 70)

    # Summary results across all T values
    summary_rows = []

    for T in args.T_values:
        print(f"\n{'─' * 70}")
        print(f"  TIMESTEPS T = {T}")
        print(f"{'─' * 70}")

        for run in range(args.num_runs):
            print(f"\n  Run {run + 1}/{args.num_runs}")

            # Fresh model + optimizer for each run
            model = SpikingCNN(beta=args.beta).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = nn.CrossEntropyLoss()

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if run == 0:
                print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

            # Per-epoch results for this run
            epoch_rows = []

            # GPU warmup (1 batch) to avoid cold-start timing artifacts
            if device.type == "cuda":
                warmup_data, warmup_target = next(iter(train_loader))
                warmup_data = warmup_data.to(device)
                _ = model(warmup_data, T)
                torch.cuda.synchronize()

            for epoch in range(1, args.epochs + 1):
                # Train
                train_stats = train_one_epoch(model, train_loader, optimizer,
                                              loss_fn, device, T)

                # Evaluate
                test_acc, inference_time = evaluate(model, test_loader, device, T)

                row = {
                    "T": T,
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

            # Save per-epoch CSV for this run
            csv_path = os.path.join(args.results_dir,
                                    f"snntorch_timing_T{T}_run{run + 1}.csv")
            save_csv(csv_path, epoch_rows)
            print(f"  Saved: {csv_path}")

            # Record summary for final epoch of this run
            final = epoch_rows[-1]
            final["total_params"] = total_params
            summary_rows.append(final)

    # Save summary CSV
    summary_path = os.path.join(args.results_dir, "snntorch_summary.csv")
    save_csv(summary_path, summary_rows)
    print(f"\n{'=' * 70}")
    print(f"Summary saved: {summary_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'T':>4} {'Run':>4} {'Epoch Time (s)':>15} {'Fwd (s)':>10} "
          f"{'Bwd (s)':>10} {'Accuracy':>10}")
    print("─" * 60)
    for row in summary_rows:
        print(f"{row['T']:>4} {row['run']:>4} {row['total_epoch_time_sec']:>15.2f} "
              f"{row['forward_time_sec']:>10.2f} {row['backward_time_sec']:>10.2f} "
              f"{row['test_accuracy_pct']:>9.2f}%")


def save_csv(path, rows):
    """Write list of dicts to CSV."""
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
