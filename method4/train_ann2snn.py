"""
=============================================================================
Method 4: ANN-to-SNN Conversion (Rate-Based Threshold Balancing)
=============================================================================
Task:       Image classification on MNIST (10 classes)
Approach:   1. Train a standard ReLU CNN to convergence (same arch as m1/m2/m3)
            2. Convert it to an SNN using SpikingJelly's ann2snn module
            3. Measure INFERENCE ONLY at various T values (64-1024)

PURPOSE:
    ANN-to-SNN conversion achieves near-lossless accuracy but requires
    many timesteps (T >= 256) for firing rates to stabilize. This method
    upper-bounds accuracy while lower-bounding latency efficiency.
    Comparing against direct training (m1/m2) at matched T values shows
    the accuracy-latency tradeoff between the two paradigms.

KEY DIFFERENCE FROM METHODS 1 & 2:
    - NO surrogate gradient training of the SNN
    - The SNN weights come directly from a pre-trained ANN
    - Only inference is measured (not training)
    - Much higher T values needed (64, 128, 256, 512, 1024)

Usage:
    python train_ann2snn.py --device cuda:0

Outputs:
    results/ann2snn_timing_T{T}_run{R}.csv  - per-T inference data
    results/ann2snn_summary.csv             - summary across all T values
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

from spikingjelly.activation_based import ann2snn, neuron, functional


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Method 4: ANN-to-SNN Conversion")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ann_epochs", type=int, default=10,
                        help="Epochs to train the source ANN")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--T_values", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256, 512, 1024],
                        help="Timestep values for SNN inference")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Repeated runs per T for variance")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET (identical to all other methods)
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size):
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
# 3. SOURCE ANN MODEL (ReLU — will be converted to SNN)
# ─────────────────────────────────────────────────────────────────────────────
class SourceANN(nn.Module):
    """
    Standard ReLU CNN for ANN-to-SNN conversion.

    Architecture matches Methods 1-3:
        Conv2d(1, 12, 5) → ReLU → AvgPool(2)    ← AvgPool instead of MaxPool
        Conv2d(12, 32, 5) → ReLU → AvgPool(2)
        Flatten → Linear(32*4*4, 800) → ReLU
        Linear(800, 10)

    NOTE: AvgPool is used instead of MaxPool because ANN-to-SNN conversion
    requires all operations to be rate-codable. MaxPool is not compatible
    with rate-based conversion (it selects max activation, which doesn't
    translate to spike rates). This is standard practice in conversion papers.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 12, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(12, 32, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 800),
            nn.ReLU(),

            nn.Linear(800, 10),
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIMING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
class GPUTimer:
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
# 5. TRAIN THE SOURCE ANN
# ─────────────────────────────────────────────────────────────────────────────
def train_source_ann(model, train_loader, test_loader, device, epochs, lr):
    """Train the ReLU ANN to convergence before conversion."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print("  Training source ANN...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += (output.argmax(1) == target).sum().item()
                total += target.size(0)
        acc = 100.0 * correct / total

        print(f"    Epoch {epoch:2d}/{epochs} │ Loss: {epoch_loss / n:.4f} │ Acc: {acc:.2f}%")

    print(f"  Source ANN trained: {acc:.2f}% accuracy")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONVERT ANN TO SNN
# ─────────────────────────────────────────────────────────────────────────────
def convert_ann_to_snn(ann_model, train_loader, device):
    """
    Convert trained ReLU ANN to SNN using SpikingJelly's ann2snn converter.

    The converter:
    1. Analyzes activation distributions using calibration data
    2. Replaces ReLU with IF (Integrate-and-Fire) neurons
    3. Normalizes weights/thresholds so firing rates match ReLU activations

    This is rate-based conversion: at large T, the average firing rate
    of each IF neuron approximates the original ReLU output.
    """
    print("  Converting ANN to SNN...")

    # Use a subset of training data for calibration
    converter = ann2snn.Converter(mode='max', dataloader=train_loader)
    snn_model = converter(ann_model)

    # NOTE: Do NOT set multi-step mode. The ann2snn converter uses torch.fx
    # tracing, which produces standard nn.Conv2d/nn.Linear layers (not
    # SpikingJelly's layer.Conv2d). These only accept 4D input (batch, C, H, W),
    # so we must loop over timesteps manually in single-step mode.

    print("  Conversion complete.")
    return snn_model


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE CONVERTED SNN AT VARIOUS T
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_snn(snn_model, test_loader, device, T):
    """
    Evaluate converted SNN inference accuracy and timing at a given T.

    At each timestep, the same input image is presented. The IF neurons
    accumulate charge and fire at rates proportional to the original
    ReLU activations. With more timesteps, the rates converge and
    accuracy approaches the original ANN's.

    NOTE: The converted model uses standard nn.Conv2d (from torch.fx tracing),
    so we must loop over T timesteps manually with 4D input per step.
    """
    snn_model.eval()
    snn_model.to(device)

    correct = 0
    total = 0
    timer = GPUTimer(device)

    timer.start()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Loop over T timesteps with standard 4D input (batch, C, H, W)
            # Accumulate output spikes/potentials across timesteps
            out_sum = torch.zeros(data.size(0), 10, device=device)
            for step in range(T):
                out = snn_model(data)  # (batch, 10) — single step
                out_sum += out

            predicted = out_sum.argmax(dim=1)

            correct += (predicted == target).sum().item()
            total += target.size(0)

            # Reset neuron states for next batch
            functional.reset_net(snn_model)

    inference_time = timer.stop()
    accuracy = 100.0 * correct / total

    return accuracy, inference_time


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(args):
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"MNIST loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    print(f"T values for SNN inference: {args.T_values}")
    print(f"Runs per T: {args.num_runs}")
    print("=" * 70)

    summary_rows = []

    for run in range(args.num_runs):
        print(f"\n{'─' * 70}")
        print(f"  Run {run + 1}/{args.num_runs}")
        print(f"{'─' * 70}")

        # Step 1: Train fresh ANN
        ann_model = SourceANN()
        total_params = sum(p.numel() for p in ann_model.parameters())
        if run == 0:
            print(f"  Model parameters: {total_params:,}")
        ann_acc = train_source_ann(ann_model, train_loader, test_loader,
                                   device, args.ann_epochs, args.lr)

        # Step 2: Convert to SNN
        ann_model.cpu()  # converter works on CPU
        snn_model = convert_ann_to_snn(ann_model, train_loader, device)

        # Step 3: Evaluate at each T value
        print(f"\n  Evaluating converted SNN at various T values...")
        epoch_rows = []

        for T in args.T_values:
            acc, inf_time = evaluate_snn(snn_model, test_loader, device, T)

            row = {
                "T": T,
                "run": run + 1,
                "ann_accuracy_pct": round(ann_acc, 2),
                "snn_accuracy_pct": round(acc, 2),
                "accuracy_drop_pct": round(ann_acc - acc, 2),
                "inference_time_sec": round(inf_time, 4),
                "inference_per_sample_ms": round(inf_time / len(test_loader.dataset) * 1000, 4),
                "total_params": total_params,
            }
            epoch_rows.append(row)
            summary_rows.append(row)

            print(f"    T={T:>5} │ SNN Acc: {acc:.2f}% │ "
                  f"ANN Acc: {ann_acc:.2f}% │ "
                  f"Drop: {ann_acc - acc:.2f}% │ "
                  f"Inf time: {inf_time:.2f}s")

        # Save per-run CSV
        csv_path = os.path.join(args.results_dir, f"ann2snn_timing_run{run + 1}.csv")
        save_csv(csv_path, epoch_rows)
        print(f"  Saved: {csv_path}")

    # Save summary
    summary_path = os.path.join(args.results_dir, "ann2snn_summary.csv")
    save_csv(summary_path, summary_rows)
    print(f"\n{'=' * 70}")
    print(f"Summary saved: {summary_path}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'T':>6} {'Run':>4} {'ANN Acc':>10} {'SNN Acc':>10} "
          f"{'Drop':>8} {'Inf Time':>10}")
    print("─" * 55)
    for row in summary_rows:
        print(f"{row['T']:>6} {row['run']:>4} {row['ann_accuracy_pct']:>9.2f}% "
              f"{row['snn_accuracy_pct']:>9.2f}% {row['accuracy_drop_pct']:>7.2f}% "
              f"{row['inference_time_sec']:>9.2f}s")


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)