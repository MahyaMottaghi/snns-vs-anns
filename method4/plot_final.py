"""
=============================================================================
FINAL COMPREHENSIVE COMPARISON — All 4 Methods
=============================================================================
Reads results from m1/, m2/, m3/, m4/ and generates:

  1.  all_accuracy_vs_T.png            — Accuracy vs T (M1, M2, M4 + M3 baseline)
  2.  all_training_time_vs_T.png       — Training time per epoch (M1, M2 + M3)
  3.  all_inference_time_vs_T.png      — Inference time (all 4 methods)
  4.  speedup_m2_over_m1.png           — SpikingJelly speedup bars
  5.  all_time_breakdown.png           — Forward/backward/data stacked bars
  6.  ann2snn_convergence.png          — M4 accuracy approaching M3
  7.  all_loss_curves.png              — Training loss over epochs
  8.  energy_estimation.png            — Theoretical neuromorphic vs GPU energy
  9.  summary_table.png + .csv         — Publication-ready summary

Usage from m4/:
    python plot_final.py

Or with custom paths:
    python plot_final.py --m1_dir /work/ofx323/hpml/project/m1/results \
                         --m2_dir /work/ofx323/hpml/project/m2/results \
                         --m3_dir /work/ofx323/hpml/project/m3/results \
                         --m4_dir ./results \
                         --plots_dir ./plots
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
    parser.add_argument("--m1_dir", type=str,
                        default="/work/ofx323/hpml/project/m1/results")
    parser.add_argument("--m2_dir", type=str,
                        default="/work/ofx323/hpml/project/m2/results")
    parser.add_argument("--m3_dir", type=str,
                        default="/work/ofx323/hpml/project/m3/results")
    parser.add_argument("--m4_dir", type=str, default="./results")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    return parser.parse_args()


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_summary(d, prefix):
    p = os.path.join(d, f"{prefix}_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

def load_epochs(d, prefix):
    files = sorted(glob.glob(os.path.join(d, f"{prefix}_timing_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


# ─── Colors & Labels ─────────────────────────────────────────────────────────

C = {"m1": "#2196F3", "m2": "#E91E63", "m3": "#4CAF50", "m4": "#FF9800"}
L = {"m1": "SNNTorch (M1)", "m2": "SpikingJelly (M2)",
     "m3": "ANN Baseline (M3)", "m4": "ANN→SNN (M4)"}


# ─── PLOT 1: Accuracy vs T ───────────────────────────────────────────────────

def plot_accuracy_vs_T(m1, m2, m3, m4, pdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, df, col in [("m1", m1, "test_accuracy_pct"),
                          ("m2", m2, "test_accuracy_pct"),
                          ("m4", m4, "snn_accuracy_pct")]:
        if df is not None:
            g = df.groupby("T")[col]
            m, s = g.mean(), g.std().fillna(0)
            ax.errorbar(m.index, m.values, yerr=s.values,
                        marker='o', capsize=4, linewidth=2, markersize=7,
                        color=C[key], label=L[key])

    if m3 is not None:
        ann_m = m3["test_accuracy_pct"].mean()
        ax.axhline(y=ann_m, color=C["m3"], linewidth=2, linestyle='--',
                   label=f'{L["m3"]} ({ann_m:.1f}%)')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("All Methods: Accuracy vs Timesteps", fontsize=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "all_accuracy_vs_T.png"), dpi=150)
    print("Saved: all_accuracy_vs_T.png")
    plt.close()


# ─── PLOT 2: Training Time vs T ──────────────────────────────────────────────

def plot_training_time_vs_T(m1, m2, m3, pdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, df in [("m1", m1), ("m2", m2)]:
        if df is not None:
            g = df.groupby("T")["total_epoch_time_sec"]
            m, s = g.mean(), g.std().fillna(0)
            ax.errorbar(m.index, m.values, yerr=s.values,
                        marker='o', capsize=4, linewidth=2, markersize=7,
                        color=C[key], label=L[key])

    if m3 is not None:
        ann_m = m3["total_epoch_time_sec"].mean()
        ax.axhline(y=ann_m, color=C["m3"], linewidth=2, linestyle='--',
                   label=f'{L["m3"]} ({ann_m:.1f}s)')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Training Time per Epoch (seconds)", fontsize=13)
    ax.set_title("Training Time vs Timesteps", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "all_training_time_vs_T.png"), dpi=150)
    print("Saved: all_training_time_vs_T.png")
    plt.close()


# ─── PLOT 3: Inference Time vs T ─────────────────────────────────────────────

def plot_inference_time_vs_T(m1, m2, m3, m4, pdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, df, col in [("m1", m1, "inference_time_sec"),
                          ("m2", m2, "inference_time_sec"),
                          ("m4", m4, "inference_time_sec")]:
        if df is not None:
            g = df.groupby("T")[col]
            m, s = g.mean(), g.std().fillna(0)
            ax.errorbar(m.index, m.values, yerr=s.values,
                        marker='o', capsize=4, linewidth=2, markersize=7,
                        color=C[key], label=L[key])

    if m3 is not None:
        ann_m = m3["inference_time_sec"].mean()
        ax.axhline(y=ann_m, color=C["m3"], linewidth=2, linestyle='--',
                   label=f'{L["m3"]} ({ann_m:.2f}s)')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Inference Time — Full Test Set (seconds)", fontsize=13)
    ax.set_title("Inference Time vs Timesteps — All Methods", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "all_inference_time_vs_T.png"), dpi=150)
    print("Saved: all_inference_time_vs_T.png")
    plt.close()


# ─── PLOT 4: Speedup ─────────────────────────────────────────────────────────

def plot_speedup(m1, m2, pdir):
    if m1 is None or m2 is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    t1 = m1.groupby("T")["total_epoch_time_sec"].mean()
    t2 = m2.groupby("T")["total_epoch_time_sec"].mean()
    common = t1.index.intersection(t2.index)
    sp = t1[common] / t2[common]

    ax.bar(range(len(common)), sp.values, color=C["m2"], alpha=0.85)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)

    for i, v in enumerate(sp.values):
        ax.text(i, v + 0.03, f"{v:.2f}×", ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel("Timesteps (T)", fontsize=12)
    ax.set_ylabel("Speedup (M1 time / M2 time)", fontsize=12)
    ax.set_title("SpikingJelly Speedup over SNNTorch", fontsize=14)
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels(common)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "speedup_m2_over_m1.png"), dpi=150)
    print("Saved: speedup_m2_over_m1.png")
    plt.close()


# ─── PLOT 5: Time Breakdown ──────────────────────────────────────────────────

def plot_time_breakdown(m1, m2, m3, pdir):
    panels = []
    if m1 is not None: panels.append(("m1", m1))
    if m2 is not None: panels.append(("m2", m2))
    if not panels:
        return

    ncols = len(panels) + (1 if m3 is not None else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
    if ncols == 1: axes = [axes]

    for idx, (key, df) in enumerate(panels):
        ax = axes[idx]
        g = df.groupby("T").agg({
            "forward_time_sec": "mean", "backward_time_sec": "mean",
            "data_loading_time_sec": "mean"
        })
        T_vals = g.index.values
        x = np.arange(len(T_vals))
        fwd, bwd, dl = (g["forward_time_sec"].values,
                        g["backward_time_sec"].values,
                        g["data_loading_time_sec"].values)
        ax.bar(x, fwd, 0.5, label="Forward", color=C[key])
        ax.bar(x, bwd, 0.5, bottom=fwd, label="Backward", color='#FF9800')
        ax.bar(x, dl, 0.5, bottom=fwd + bwd, label="Data Load", color='#9E9E9E')
        ax.set_xlabel("T"); ax.set_title(L[key])
        ax.set_xticks(x); ax.set_xticklabels(T_vals)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    if m3 is not None:
        ax = axes[-1]
        fwd = m3["forward_time_sec"].mean()
        bwd = m3["backward_time_sec"].mean()
        dl = m3["data_loading_time_sec"].mean()
        ax.bar([0], [fwd], 0.5, label="Forward", color=C["m3"])
        ax.bar([0], [bwd], 0.5, bottom=[fwd], label="Backward", color='#FF9800')
        ax.bar([0], [dl], 0.5, bottom=[fwd + bwd], label="Data Load", color='#9E9E9E')
        ax.set_title(L["m3"]); ax.set_xticks([0]); ax.set_xticklabels(["T=1"])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel("Time (seconds)")
    fig.suptitle("Epoch Time Breakdown — All Methods", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "all_time_breakdown.png"), dpi=150,
                bbox_inches='tight')
    print("Saved: all_time_breakdown.png")
    plt.close()


# ─── PLOT 6: ANN-to-SNN Convergence ──────────────────────────────────────────

def plot_ann2snn_convergence(m3, m4, pdir):
    if m4 is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))

    g = m4.groupby("T")["snn_accuracy_pct"]
    m, s = g.mean(), g.std().fillna(0)
    ax.errorbar(m.index, m.values, yerr=s.values,
                marker='s', capsize=4, linewidth=2, markersize=7,
                color=C["m4"], label="Converted SNN (M4)")

    # Source ANN from M4 data
    ann_acc = m4["ann_accuracy_pct"].mean()
    ax.axhline(y=ann_acc, color=C["m3"], linewidth=2, linestyle='--',
               label=f'Source ANN ({ann_acc:.1f}%)')

    # M3 baseline if available
    if m3 is not None:
        m3_acc = m3["test_accuracy_pct"].mean()
        ax.axhline(y=m3_acc, color=C["m3"], linewidth=1, linestyle=':',
                   alpha=0.5, label=f'ANN Baseline M3 ({m3_acc:.1f}%)')

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("ANN-to-SNN Conversion: Accuracy Convergence", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "ann2snn_convergence.png"), dpi=150)
    print("Saved: ann2snn_convergence.png")
    plt.close()


# ─── PLOT 7: Loss Curves ─────────────────────────────────────────────────────

def plot_loss_curves(m1_ep, m2_ep, m3_ep, pdir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for ep, key, T_val in [(m1_ep, "m1", 16), (m2_ep, "m2", 16)]:
        if ep is not None and T_val in ep["T"].values:
            sub = ep[ep["T"] == T_val]
            g = sub.groupby("epoch")["train_loss"]
            m, s = g.mean(), g.std().fillna(0)
            ax.plot(m.index, m.values, marker='o', markersize=4, linewidth=2,
                    color=C[key], label=f'{L[key]} (T={T_val})')
            ax.fill_between(m.index, m - s, m + s, alpha=0.12, color=C[key])

    if m3_ep is not None:
        g = m3_ep.groupby("epoch")["train_loss"]
        m, s = g.mean(), g.std().fillna(0)
        ax.plot(m.index, m.values, marker='s', markersize=4, linewidth=2,
                color=C["m3"], linestyle='--', label=L["m3"])
        ax.fill_between(m.index, m - s, m + s, alpha=0.12, color=C["m3"])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Curves (SNN methods at T=16)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "all_loss_curves.png"), dpi=150)
    print("Saved: all_loss_curves.png")
    plt.close()


# ─── PLOT 8: Energy Estimation ───────────────────────────────────────────────

def plot_energy_estimation(m1, m2, m3, m4, pdir):
    """
    Theoretical energy estimation comparing neuromorphic vs GPU execution.

    Energy model (Dampfhoffer et al., 2022; 45nm technology):
      - ANN MAC operation:  E_MAC = 4.6 pJ
      - SNN AC operation:   E_AC  = 0.9 pJ

    Architecture MACs per forward pass (single image):
      Conv1: 12 × 24 × 24 × 1 × 5 × 5  = 172,800
      Conv2: 32 × 8  × 8  × 12 × 5 × 5 = 614,400
      FC1:   512 × 800                   = 409,600
      FC2:   800 × 10                    = 8,000
      Total MACs per pass                = 1,204,800

    GPU: executes full MACs regardless of spikes (no sparsity benefit)
      ANN: 1 × MACs
      SNN: T × MACs (repeats for each timestep)

    Neuromorphic (theoretical): SNN only computes when spikes arrive
      SNN: T × MACs × spike_rate × (E_AC / E_MAC)
      Assuming average spike rate ~0.3 (30% of neurons fire per step)
    """
    MACS_PER_PASS = 1_204_800
    E_MAC = 4.6  # pJ
    E_AC = 0.9   # pJ
    SPIKE_RATE = 0.3  # conservative estimate

    T_values = [4, 8, 16, 32, 64]

    # GPU energy (proportional to compute, same ops for ANN and SNN)
    ann_gpu = MACS_PER_PASS * E_MAC  # pJ per image
    snn_gpu = [T * MACS_PER_PASS * E_MAC for T in T_values]

    # Neuromorphic energy (theoretical — SNN benefits from sparsity)
    ann_neuro = MACS_PER_PASS * E_MAC  # ANN still does MACs
    snn_neuro = [T * MACS_PER_PASS * SPIKE_RATE * E_AC for T in T_values]

    # Convert to microjoules for readability
    ann_gpu_uj = ann_gpu / 1e6
    snn_gpu_uj = [e / 1e6 for e in snn_gpu]
    ann_neuro_uj = ann_neuro / 1e6
    snn_neuro_uj = [e / 1e6 for e in snn_neuro]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: GPU execution
    x = np.arange(len(T_values))
    width = 0.35
    ax1.bar(x - width/2, snn_gpu_uj, width, label='SNN on GPU', color=C["m1"])
    ax1.axhline(y=ann_gpu_uj, color=C["m3"], linewidth=2, linestyle='--',
                label=f'ANN on GPU ({ann_gpu_uj:.2f} μJ)')
    ax1.set_xlabel("Timesteps (T)", fontsize=12)
    ax1.set_ylabel("Energy per Image (μJ)", fontsize=12)
    ax1.set_title("GPU: SNN is T× MORE expensive", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(T_values)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add T× labels
    for i, (t, e) in enumerate(zip(T_values, snn_gpu_uj)):
        ratio = e / ann_gpu_uj
        ax1.text(i - width/2, e + 0.1, f"{ratio:.0f}×", ha='center',
                 fontsize=10, fontweight='bold', color='red')

    # Right: Neuromorphic execution (theoretical)
    ax2.bar(x - width/2, snn_neuro_uj, width, label='SNN on Neuromorphic',
            color=C["m2"])
    ax2.axhline(y=ann_neuro_uj, color=C["m3"], linewidth=2, linestyle='--',
                label=f'ANN on GPU ({ann_neuro_uj:.2f} μJ)')
    ax2.set_xlabel("Timesteps (T)", fontsize=12)
    ax2.set_ylabel("Energy per Image (μJ)", fontsize=12)
    ax2.set_title("Neuromorphic: SNN CAN be cheaper (theoretical)", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(T_values)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add savings labels
    for i, (t, e) in enumerate(zip(T_values, snn_neuro_uj)):
        ratio = ann_neuro_uj / e if e > 0 else 0
        if ratio > 1:
            ax2.text(i - width/2, e + 0.05, f"{ratio:.1f}× cheaper",
                     ha='center', fontsize=9, fontweight='bold', color='green')
        else:
            ax2.text(i - width/2, e + 0.05, f"{1/ratio:.1f}× costlier",
                     ha='center', fontsize=9, fontweight='bold', color='red')

    fig.suptitle("Energy Estimation: GPU vs Neuromorphic Hardware\n"
                 "(45nm technology, E_MAC=4.6pJ, E_AC=0.9pJ, spike rate=30%)",
                 fontsize=14, y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "energy_estimation.png"), dpi=150,
                bbox_inches='tight')
    print("Saved: energy_estimation.png")
    plt.close()


# ─── PLOT 9: Summary Table ───────────────────────────────────────────────────

def make_summary_table(m1, m2, m3, m4, pdir):
    rows = []

    if m3 is not None:
        rows.append({
            "Method": "ANN Baseline (M3)",
            "Framework": "PyTorch",
            "Best Acc (%)": f"{m3['test_accuracy_pct'].max():.2f}",
            "T at Best": "1",
            "Epoch Time (s)": f"{m3['total_epoch_time_sec'].mean():.2f}",
            "Fwd (s)": f"{m3['forward_time_sec'].mean():.2f}",
            "Bwd (s)": f"{m3['backward_time_sec'].mean():.2f}",
            "Inference (s)": f"{m3['inference_time_sec'].mean():.2f}",
            "Params": f"{int(m3['total_params'].iloc[0]):,}",
        })

    if m1 is not None:
        best_T = m1.groupby("T")["test_accuracy_pct"].mean().idxmax()
        br = m1[m1["T"] == best_T]
        rows.append({
            "Method": "SNNTorch (M1)",
            "Framework": "SNNTorch",
            "Best Acc (%)": f"{br['test_accuracy_pct'].max():.2f}",
            "T at Best": str(best_T),
            "Epoch Time (s)": f"{br['total_epoch_time_sec'].mean():.2f}",
            "Fwd (s)": f"{br['forward_time_sec'].mean():.2f}",
            "Bwd (s)": f"{br['backward_time_sec'].mean():.2f}",
            "Inference (s)": f"{br['inference_time_sec'].mean():.2f}",
            "Params": f"{int(br['total_params'].iloc[0]):,}",
        })

    if m2 is not None:
        best_T = m2.groupby("T")["test_accuracy_pct"].mean().idxmax()
        br = m2[m2["T"] == best_T]
        rows.append({
            "Method": "SpikingJelly (M2)",
            "Framework": "SpikingJelly",
            "Best Acc (%)": f"{br['test_accuracy_pct'].max():.2f}",
            "T at Best": str(best_T),
            "Epoch Time (s)": f"{br['total_epoch_time_sec'].mean():.2f}",
            "Fwd (s)": f"{br['forward_time_sec'].mean():.2f}",
            "Bwd (s)": f"{br['backward_time_sec'].mean():.2f}",
            "Inference (s)": f"{br['inference_time_sec'].mean():.2f}",
            "Params": f"{int(br['total_params'].iloc[0]):,}",
        })

    if m4 is not None:
        best_T = m4.groupby("T")["snn_accuracy_pct"].mean().idxmax()
        br = m4[m4["T"] == best_T]
        rows.append({
            "Method": "ANN→SNN (M4)",
            "Framework": "SpikingJelly ann2snn",
            "Best Acc (%)": f"{br['snn_accuracy_pct'].max():.2f}",
            "T at Best": str(best_T),
            "Epoch Time (s)": "N/A",
            "Fwd (s)": "N/A",
            "Bwd (s)": "N/A",
            "Inference (s)": f"{br['inference_time_sec'].mean():.2f}",
            "Params": f"{int(br['total_params'].iloc[0]):,}",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(pdir, "summary_table.csv"), index=False)
    print("Saved: summary_table.csv")

    fig, ax = plt.subplots(figsize=(16, 2 + len(rows) * 0.6))
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, len(rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("Performance Summary — All Methods", fontsize=14,
                 fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(pdir, "summary_table.png"), dpi=150,
                bbox_inches='tight')
    print("Saved: summary_table.png")
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    m1 = load_summary(args.m1_dir, "snntorch")
    m2 = load_summary(args.m2_dir, "spikingjelly")
    m3 = load_summary(args.m3_dir, "ann")
    m4 = load_summary(args.m4_dir, "ann2snn")

    m1_ep = load_epochs(args.m1_dir, "snntorch")
    m2_ep = load_epochs(args.m2_dir, "spikingjelly")
    m3_ep = load_epochs(args.m3_dir, "ann")

    found = []
    for name, d in [("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4)]:
        if d is not None: found.append(name)
    print(f"Found: {', '.join(found)}")

    if not found:
        print("No results found!")
        return

    print()
    plot_accuracy_vs_T(m1, m2, m3, m4, args.plots_dir)
    plot_training_time_vs_T(m1, m2, m3, args.plots_dir)
    plot_inference_time_vs_T(m1, m2, m3, m4, args.plots_dir)
    plot_speedup(m1, m2, args.plots_dir)
    plot_time_breakdown(m1, m2, m3, args.plots_dir)
    plot_ann2snn_convergence(m3, m4, args.plots_dir)
    plot_loss_curves(m1_ep, m2_ep, m3_ep, args.plots_dir)
    plot_energy_estimation(m1, m2, m3, m4, args.plots_dir)
    make_summary_table(m1, m2, m3, m4, args.plots_dir)

    print(f"\n{'=' * 50}")
    print(f"All 9 plots + summary saved to {args.plots_dir}/")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
