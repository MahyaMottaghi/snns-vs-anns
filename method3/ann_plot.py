"""
=============================================================================
Plotting script for Method 3 (ANN Baseline) results
=============================================================================
Reads CSV files from ./results/ and generates ANN-only performance plots.

Plots generated:
  1. ann_loss_curve.png        — Training loss over epochs
  2. ann_accuracy_curve.png    — Test accuracy over epochs
  3. ann_time_breakdown.png    — Forward / backward / data loading bar
  4. ann_summary.png           — Summary table as image

Usage:
    python ann_plot.py
    python ann_plot.py --results_dir ./results --plots_dir ./plots
=============================================================================
"""

import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    return parser.parse_args()


def load_results(results_dir):
    """Load all per-epoch CSVs and summary."""
    # Load individual run files
    pattern = os.path.join(results_dir, "ann_timing_run*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found in {results_dir}")
        print("Run train_ann.py first.")
        return None, None

    all_epochs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Load summary
    summary_path = os.path.join(results_dir, "ann_summary.csv")
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    return all_epochs, summary


# ─── PLOT 1: Loss Curve ──────────────────────────────────────────────────────

def plot_loss_curve(all_epochs, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = all_epochs.groupby("epoch")["train_loss"]
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    ax.plot(means.index, means.values, marker='o', markersize=6,
            linewidth=2, color='#4CAF50', label='ANN Baseline')
    ax.fill_between(means.index, means.values - stds.values,
                    means.values + stds.values, alpha=0.15, color='#4CAF50')

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("ANN Baseline: Training Loss Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann_loss_curve.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ─── PLOT 2: Accuracy Curve ──────────────────────────────────────────────────

def plot_accuracy_curve(all_epochs, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = all_epochs.groupby("epoch")["test_accuracy_pct"]
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    ax.plot(means.index, means.values, marker='s', markersize=6,
            linewidth=2, color='#4CAF50', label='ANN Baseline')
    ax.fill_between(means.index, means.values - stds.values,
                    means.values + stds.values, alpha=0.15, color='#4CAF50')

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("ANN Baseline: Test Accuracy over Epochs", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ymin = max(0, means.min() - 5)
    ax.set_ylim(ymin, 100)
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann_accuracy_curve.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ─── PLOT 3: Time Breakdown ──────────────────────────────────────────────────

def plot_time_breakdown(summary, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 5))

    fwd = summary["forward_time_sec"].mean()
    bwd = summary["backward_time_sec"].mean()
    dl = summary["data_loading_time_sec"].mean()
    total = fwd + bwd + dl

    bars = ax.bar(["ANN Baseline"], [fwd], 0.5, label=f"Forward ({fwd:.2f}s)", color='#4CAF50')
    ax.bar(["ANN Baseline"], [bwd], 0.5, bottom=[fwd],
           label=f"Backward ({bwd:.2f}s)", color='#FF9800')
    ax.bar(["ANN Baseline"], [dl], 0.5, bottom=[fwd + bwd],
           label=f"Data Loading ({dl:.2f}s)", color='#9E9E9E')

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"ANN Baseline: Epoch Time Breakdown ({total:.2f}s total)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann_time_breakdown.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ─── PLOT 4: Summary Table ───────────────────────────────────────────────────

def plot_summary_table(all_epochs, summary, plots_dir):
    # Compute stats
    best_acc = all_epochs["test_accuracy_pct"].max()
    mean_acc = summary["test_accuracy_pct"].mean()
    std_acc = summary["test_accuracy_pct"].std()
    mean_time = summary["total_epoch_time_sec"].mean()
    std_time = summary["total_epoch_time_sec"].std()
    mean_fwd = summary["forward_time_sec"].mean()
    mean_bwd = summary["backward_time_sec"].mean()
    mean_inf = summary["inference_time_sec"].mean()
    params = int(summary["total_params"].iloc[0])
    num_runs = len(summary)

    rows = [
        ["Best Test Accuracy", f"{best_acc:.2f}%"],
        ["Final Accuracy (mean ± std)", f"{mean_acc:.2f} ± {std_acc:.2f}%"],
        ["Epoch Time (mean ± std)", f"{mean_time:.2f} ± {std_time:.2f}s"],
        ["Forward Pass Time", f"{mean_fwd:.2f}s"],
        ["Backward Pass Time", f"{mean_bwd:.2f}s"],
        ["Inference Time (full test set)", f"{mean_inf:.2f}s"],
        ["Trainable Parameters", f"{params:,}"],
        ["Timesteps (T)", "1 (no temporal loop)"],
        ["Number of Runs", str(num_runs)],
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc='left',
        loc='center',
        colWidths=[0.5, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(2):
            table[i, j].set_facecolor(color)

    ax.set_title("ANN Baseline (Method 3) — Performance Summary",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    all_epochs, summary = load_results(args.results_dir)
    if all_epochs is None:
        return

    print(f"Loaded {len(all_epochs)} epoch records from {len(summary)} runs")
    print()

    plot_loss_curve(all_epochs, args.plots_dir)
    plot_accuracy_curve(all_epochs, args.plots_dir)

    if summary is not None:
        plot_time_breakdown(summary, args.plots_dir)
        plot_summary_table(all_epochs, summary, args.plots_dir)

    print(f"\nAll plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
