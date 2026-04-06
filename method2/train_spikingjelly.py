"""
=============================================================================
Method 2: Event-Driven SNN Computation (SpikingJelly)
=============================================================================
Task:       Image classification on MNIST (10 classes)
Model:      Convolutional SNN with LIF neurons
            Architecture: 12C5 - MP2 - 32C5 - MP2 - FC800 - FC10
            (IDENTICAL to Method 1 / SNNTorch for fair comparison)
Framework:  SpikingJelly (PyTorch-based, with multi-step CUDA acceleration)
Surrogate:  Arctangent (ATan) — same as Method 1

KEY DIFFERENCE FROM METHOD 1:
    SNNTorch loops over T timesteps in Python (one kernel launch per step).
    SpikingJelly's multi-step mode fuses the entire T-step simulation into
    fewer CUDA kernel calls, reducing Python-level overhead and GPU kernel
    launch latency. This is the core parallelization advantage being measured.

Usage:
    python train_spikingjelly.py --device cuda:0
    python train_spikingjelly.py --device cpu

Outputs:
    results/spikingjelly_timing_T{T}_run{R}.csv  - per-epoch timing data
    results/spikingjelly_summary.csv             - summary across all T values
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

from spikingjelly.activation_based import neuron, layer, surrogate, functional

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Method 2: SpikingJelly MNIST Training")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to train on (cuda:0 or cpu)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per T value")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="LIF neuron time constant (equivalent to beta=1-1/tau)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download/store MNIST")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory to save timing results")
    parser.add_argument("--T_values", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64],
                        help="List of timestep values to sweep")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of repeated runs per T for variance")
    parser.add_argument("--backend", type=str, default="torch",
                        choices=["torch", "cupy"],
                        help="SpikingJelly neuron backend (torch or cupy)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET (identical to Method 1)
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size):
    """
    Load MNIST with standard normalization.
    IDENTICAL preprocessing to Method 1 for fair comparison.
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
    Convolutional SNN for MNIST classification using SpikingJelly.

    Architecture (IDENTICAL to Method 1):
        Conv2d(1, 12, 5)  → LIF → MaxPool(2)
        Conv2d(12, 32, 5) → LIF → MaxPool(2)
        Flatten → Linear(32*4*4, 800) → LIF
        Linear(800, 10) → LIF (output)

    KEY DIFFERENCE: All layers are set to step_mode='m' (multi-step).
    In multi-step mode, SpikingJelly processes ALL T timesteps in a single
    call rather than looping in Python. For neuron layers with cupy/triton
    backend, the entire T-step LIF simulation is fused into one CUDA kernel.

    Communication pattern:
        - Multi-step mode: T timesteps processed in a fused operation
          (reduced kernel launch overhead compared to Method 1)
        - Parallel across batch dimension (same as Method 1)
        - Within each timestep: standard CNN forward pass
    """

    def __init__(self, tau=2.0, backend="torch"):
        super().__init__()

        # NOTE: tau=2.0 corresponds to beta=0.5 in SNNTorch (beta = 1 - 1/tau)
        # For tau to match Method 1's beta=0.95, use tau = 1/(1-0.95) = 20.0
        # We use tau=20.0 to match Method 1's beta=0.95

        # Use SpikingJelly's layer wrappers for multi-step compatibility
        self.network = nn.Sequential(
            # Layer 1: Conv → LIF → Pool
            layer.Conv2d(1, 12, 5),         # 28x28 → 24x24
            neuron.LIFNode(tau=tau,
                           surrogate_function=surrogate.ATan(),
                           backend=backend),
            layer.MaxPool2d(2),              # 24x24 → 12x12

            # Layer 2: Conv → LIF → Pool
            layer.Conv2d(12, 32, 5),        # 12x12 → 8x8
            neuron.LIFNode(tau=tau,
                           surrogate_function=surrogate.ATan(),
                           backend=backend),
            layer.MaxPool2d(2),              # 8x8   → 4x4

            # Flatten
            layer.Flatten(),

            # Layer 3: FC → LIF
            layer.Linear(32 * 4 * 4, 800),
            neuron.LIFNode(tau=tau,
                           surrogate_function=surrogate.ATan(),
                           backend=backend),

            # Layer 4 (output): FC → LIF
            layer.Linear(800, 10),
            neuron.LIFNode(tau=tau,
                           surrogate_function=surrogate.ATan(),
                           backend=backend),
        )

        # Set all layers to multi-step mode
        # This is the KEY optimization: instead of looping T times in Python,
        # SpikingJelly processes all T steps in a single fused call
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        """
        Forward pass — multi-step mode.

        Args:
            x: Input tensor of shape (T, batch, 1, 28, 28)
               Note: SpikingJelly multi-step expects time as the FIRST dim

        Returns:
            Output spikes of shape (T, batch, 10)

        NOTE: Unlike Method 1, there is NO Python for-loop over T here.
              The entire temporal simulation is handled internally by
              SpikingJelly's multi-step infrastructure, which can fuse
              operations across timesteps in a single CUDA kernel call.
        """
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIMING UTILITIES (identical to Method 1)
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
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, T):
    """
    Train for one epoch. Returns timing breakdown and loss.

    KEY DIFFERENCE from Method 1:
        - Input is expanded to (T, batch, C, H, W) by repeating the image
        - Forward pass is a SINGLE call (no Python loop)
        - functional.reset_net() clears neuron states between batches
    """
    model.train()
    timer = GPUTimer(device)

    epoch_loss = 0.0
    num_batches = 0
    total_forward = 0.0
    total_backward = 0.0
    epoch_start = time.perf_counter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)      # (batch, 1, 28, 28)
        target = target.to(device, non_blocking=True)   # (batch,)

        # Expand input to (T, batch, 1, 28, 28) — repeat same image T times
        # This matches Method 1's approach of feeding the same image each step
        data_seq = data.unsqueeze(0).repeat(T, 1, 1, 1, 1)

        # ── Forward pass timing ──
        timer.start()
        out = model(data_seq)  # (T, batch, 10) — single call, no Python loop

        # Loss: cross-entropy on summed output membrane potential
        # Sum over T timesteps, same as Method 1
        out_sum = out.sum(dim=0)  # (batch, 10)
        loss = loss_fn(out_sum, target)
        forward_time = timer.stop()
        total_forward += forward_time

        # ── Backward pass timing ──
        timer.start()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = timer.stop()
        total_backward += backward_time

        # Reset neuron states for next batch
        # (clears membrane potentials — equivalent to re-initializing in Method 1)
        functional.reset_net(model)

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
    Classification based on highest total spike count across T timesteps.
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

            # Expand to (T, batch, 1, 28, 28)
            data_seq = data.unsqueeze(0).repeat(T, 1, 1, 1, 1)

            out = model(data_seq)  # (T, batch, 10)

            # Sum spikes across timesteps
            total_spikes = out.sum(dim=0)  # (batch, 10)
            predicted = total_spikes.argmax(dim=1)

            correct += (predicted == target).sum().item()
            total += target.size(0)

            functional.reset_net(model)

    inference_time = timer.stop()
    accuracy = 100.0 * correct / total

    return accuracy, inference_time


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Convert beta=0.95 (Method 1) to equivalent tau for SpikingJelly
    # beta = 1 - 1/tau → tau = 1/(1-beta) = 1/(1-0.95) = 20.0
    tau = 20.0  # Matches Method 1's beta=0.95
    print(f"Using device: {device}")
    print(f"Backend: {args.backend}")
    print(f"tau={tau} (equivalent to SNNTorch beta=0.95)")

    # Load data once
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"MNIST loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print(f"Batch size: {args.batch_size}, Batches/epoch: {len(train_loader)}")
    print(f"T values to sweep: {args.T_values}")
    print(f"Runs per T: {args.num_runs}")
    print("=" * 70)

    summary_rows = []

    for T in args.T_values:
        print(f"\n{'─' * 70}")
        print(f"  TIMESTEPS T = {T}")
        print(f"{'─' * 70}")

        for run in range(args.num_runs):
            print(f"\n  Run {run + 1}/{args.num_runs}")

            # Fresh model + optimizer for each run
            model = SpikingCNN(tau=tau, backend=args.backend).to(device)
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
                warmup_seq = warmup_data.unsqueeze(0).repeat(T, 1, 1, 1, 1)
                _ = model(warmup_seq)
                functional.reset_net(model)
                torch.cuda.synchronize()

            for epoch in range(1, args.epochs + 1):
                train_stats = train_one_epoch(model, train_loader, optimizer,
                                              loss_fn, device, T)
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

            # Save per-epoch CSV
            csv_path = os.path.join(args.results_dir,
                                    f"spikingjelly_timing_T{T}_run{run + 1}.csv")
            save_csv(csv_path, epoch_rows)
            print(f"  Saved: {csv_path}")

            final = epoch_rows[-1]
            final["total_params"] = total_params
            final["backend"] = args.backend
            summary_rows.append(final)

    # Save summary
    summary_path = os.path.join(args.results_dir, "spikingjelly_summary.csv")
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
