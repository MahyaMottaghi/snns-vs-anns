#!/bin/bash
#----------------------------------------------------
# Method 1: SNNTorch Training Batch Job
# Partition: gpu1v100 (1x NVIDIA Tesla V100S)
#----------------------------------------------------
#SBATCH -J snntorch_bench          # Job Name
#SBATCH -o snntorch_bench.o%j      # stdout output file (%j = job ID)
#SBATCH -e snntorch_bench.e%j      # stderr error file
#SBATCH -p gpu1v100                # GPU partition (1x V100S per node)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 1                       # Number of tasks
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH -t 04:00:00                # Max run time (4 hours)
#SBATCH --mail-type=ALL            # Email on start, end, fail
#SBATCH --mail-user=mahya.mottaghi@my.utsa.edu 

#----------------------------------------------------
# Environment setup
#----------------------------------------------------
# Load Anaconda (only available on job nodes)
module load anaconda3/2024.10-1

# Activate the conda environment created by setup_env.sh
source activate snn_bench

# Print job info for logging
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Start Time:    $(date)"
echo "Working Dir:   $(pwd)"
echo "=============================================="

# Verify GPU is accessible
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Starting SNNTorch training benchmark..."
echo ""

#----------------------------------------------------
# Run the experiment
#----------------------------------------------------
# Full experiment: T = 4, 8, 16, 32, 64 | 10 epochs | 3 runs each
python train_snntorch.py \
    --device cuda:0 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --beta 0.95 \
    --T_values 4 8 16 32 64 \
    --num_runs 3 \
    --data_dir ./data \
    --results_dir ./results

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to ./results/"
echo "=============================================="

#----------------------------------------------------
# Generate plots (optional — can also do this locally)
#----------------------------------------------------
python plot_snntorch.py --results_dir ./results --plots_dir ./plots

echo "Plots saved to ./plots/"
echo "Job finished at $(date)"
