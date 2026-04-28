"""
=============================================================================
Plotting script for Method 4 (ANN-to-SNN Conversion) results
=============================================================================
Plots generated:
  1. ann2snn_accuracy_vs_T.png      — Converted SNN accuracy approaching ANN
  2. ann2snn_accuracy_drop_vs_T.png — Accuracy drop from source ANN vs T
  3. ann2snn_inference_time_vs_T.png — Inference time scaling with T
  4. ann2snn_summary.png             — Summary table

Usage:
    python ann2snn_plot.py
    python ann2snn_plot.py --results_dir ./results --plots_dir ./plots
=============================================================================
"""

import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    return parser.parse_args()


def load_results(results_dir):
    pattern = os.path.join(results_dir, "ann2snn_timing_run*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files found in {results_dir}")
        print("Run train_ann2snn.py first.")
        return None, None

    all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    summary_path = os.path.join(results_dir, "ann2snn_summary.csv")
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else all_data

    return all_data, summary


# ─── PLOT 1: Accuracy vs T (convergence) ─────────────────────────────────────

def plot_accuracy_vs_T(data, plots_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    g = data.groupby("T")["snn_accuracy_pct"]
    m, s = g.mean(), g.std().fillna(0)

    ax.errorbar(m.index, m.values, yerr=s.values,
                marker='s', capsize=4, linewidth=2, markersize=7,
                color='#FF9800', label='Converted SNN')

    # Source ANN accuracy line
    ann_acc = data["ann_accuracy_pct"].mean()
    ax.axhline(y=ann_acc, color='#4CAF50', linewidth=2, linestyle='--',
               label=f'Source ANN ({ann_acc:.1f}%)')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("ANN-to-SNN Conversion: Accuracy Convergence", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann2snn_accuracy_vs_T.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ─── PLOT 2: Accuracy Drop vs T ──────────────────────────────────────────────

def plot_accuracy_drop(data, plots_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    g = data.groupby("T")["accuracy_drop_pct"]
    m, s = g.mean(), g.std().fillna(0)

    ax.errorbar(m.index, m.values, yerr=s.values,
                marker='o', capsize=4, linewidth=2, markersize=7,
                color='#F44336', label='Accuracy Drop')

    ax.axhline(y=0, color='gray', linewidth=1, linestyle='--', label='Zero drop (lossless)')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Accuracy Drop from Source ANN (%)", fontsize=13)
    ax.set_title("ANN-to-SNN Conversion: Accuracy Loss vs Timesteps", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann2snn_accuracy_drop_vs_T.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ─── PLOT 3: Inference Time vs T ─────────────────────────────────────────────

def plot_inference_time(data, plots_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    g = data.groupby("T")["inference_time_sec"]
    m, s = g.mean(), g.std().fillna(0)

    ax.errorbar(m.index, m.values, yerr=s.values,
                marker='D', capsize=4, linewidth=2, markersize=7,
                color='#FF9800', label='Converted SNN Inference')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Inference Time — Full Test Set (seconds)", fontsize=13)
    ax.set_title("ANN-to-SNN Conversion: Inference Time vs Timesteps", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann2snn_inference_time_vs_T.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ─── PLOT 4: Summary Table ───────────────────────────────────────────────────

def plot_summary_table(data, plots_dir):
    ann_acc = data["ann_accuracy_pct"].mean()
    best_T = data.groupby("T")["snn_accuracy_pct"].mean().idxmax()
    best_acc = data.groupby("T")["snn_accuracy_pct"].mean().max()
    best_drop = ann_acc - best_acc
    best_inf = data[data["T"] == best_T]["inference_time_sec"].mean()
    params = int(data["total_params"].iloc[0])

    # Find minimum T for < 1% accuracy drop
    g = data.groupby("T")["accuracy_drop_pct"].mean()
    near_lossless = g[g < 1.0]
    min_T_lossless = int(near_lossless.index.min()) if len(near_lossless) > 0 else "N/A"

    rows = [
        ["Source ANN Accuracy", f"{ann_acc:.2f}%"],
        ["Best Converted SNN Accuracy", f"{best_acc:.2f}%"],
        ["T at Best Accuracy", str(best_T)],
        ["Accuracy Drop at Best T", f"{best_drop:.2f}%"],
        ["Min T for <1% Drop", str(min_T_lossless)],
        ["Inference Time at Best T", f"{best_inf:.2f}s"],
        ["Parameters", f"{params:,}"],
        ["Number of Runs", str(data["run"].nunique())],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))
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

    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, len(rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(2):
            table[i, j].set_facecolor(color)

    ax.set_title("ANN-to-SNN Conversion (Method 4) — Summary",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(plots_dir, "ann2snn_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    data, summary = load_results(args.results_dir)
    if data is None:
        return

    print(f"Loaded {len(data)} records, T values: {sorted(data['T'].unique())}")
    print()

    plot_accuracy_vs_T(data, args.plots_dir)
    plot_accuracy_drop(data, args.plots_dir)
    plot_inference_time(data, args.plots_dir)
    plot_summary_table(data, args.plots_dir)

    print(f"\nAll plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
