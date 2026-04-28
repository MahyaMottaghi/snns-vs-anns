#!/bin/bash
#----------------------------------------------------
# UTSA ARC - Method 3: ANN Baseline Training Batch Job
# Partition: gpu1v100 (1x NVIDIA Tesla V100S)
#----------------------------------------------------
#SBATCH -J ann_bench                # Job Name
#SBATCH -o ann_bench.o%j            # stdout output file
#SBATCH -e ann_bench.e%j            # stderr error file
#SBATCH -p gpu1v100                 # GPU partition
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 1                        # Number of tasks
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH -t 01:00:00                 # 1 hour (ANN is much faster, no T sweep)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahya.mottaghi@my.utsa.edu 

#----------------------------------------------------
# Environment setup
#----------------------------------------------------
module load anaconda3/2024.10-1
source activate snn_bench

# Change to project directory
cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "Start Time:    $(date)"
echo "=============================================="

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Starting ANN baseline training..."
echo ""

#----------------------------------------------------
# Run the experiment
#----------------------------------------------------
python train_ann.py \
    --device cuda:0 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --num_runs 3 \
    --data_dir ./data \
    --results_dir ./results

echo ""
echo "=============================================="
echo "ANN training complete at $(date)"
echo "=============================================="

#----------------------------------------------------
# Generate master comparison plots (all 3 methods)
#----------------------------------------------------
python plot_all.py \
    --m1_dir $WORK/hpml/project/m1/results \
    --m2_dir $WORK/hpml/project/m2/results \
    --m3_dir ./results \
    --plots_dir ./plots

echo "All comparison plots saved to ./plots/"
echo "Job finished at $(date)"
