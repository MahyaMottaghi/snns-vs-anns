#!/bin/bash
#----------------------------------------------------
# UTSA ARC - Method 2: SpikingJelly Training Batch Job
# Partition: gpu1v100 (1x NVIDIA Tesla V100S)
#----------------------------------------------------
#SBATCH -J sj_bench                # Job Name
#SBATCH -o sj_bench.o%j            # stdout output file (%j = job ID)
#SBATCH -e sj_bench.e%j            # stderr error file
#SBATCH -p gpu1v100                # GPU partition (1x V100S per node)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Number of tasks
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH -t 04:00:00                # Max run time (4 hours)
#SBATCH --mail-type=ALL            # Email on start, end, fail
#SBATCH --mail-user=YOUR_EMAIL@utsa.edu  # <-- CHANGE THIS

#----------------------------------------------------
# Environment setup
#----------------------------------------------------
module load anaconda3/2024.10-1
source activate snn_bench

# Print job info
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Start Time:    $(date)"
echo "Working Dir:   $(pwd)"
echo "=============================================="

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import spikingjelly
print(f'SpikingJelly OK')
"

echo ""
echo "Starting SpikingJelly training benchmark (torch backend)..."
echo ""

#----------------------------------------------------
# Run the experiment (torch backend)
#----------------------------------------------------
python train_spikingjelly.py \
    --device cuda:0 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --T_values 4 8 16 32 64 \
    --num_runs 3 \
    --backend torch \
    --data_dir ./data \
    --results_dir ./results

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to ./results/"
echo "=============================================="

#----------------------------------------------------
# Generate plots (including comparisons if Method 1 results exist)
#----------------------------------------------------
python plot_spikingjelly.py --results_dir ./results --plots_dir ./plots

echo "Plots saved to ./plots/"
echo "Job finished at $(date)"
