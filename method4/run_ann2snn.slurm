#!/bin/bash
#----------------------------------------------------
# UTSA ARC - Method 4: ANN-to-SNN Conversion
# Partition: gpu1v100 (1x NVIDIA Tesla V100S)
#----------------------------------------------------
#SBATCH -J ann2snn_bench            # Job Name
#SBATCH -o ann2snn_bench.o%j        # stdout output file
#SBATCH -e ann2snn_bench.e%j        # stderr error file
#SBATCH -p gpu1v100                 # GPU partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00                 # 4 hours (high T inference takes time)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@utsa.edu  # <-- CHANGE THIS

module load anaconda3/2024.10-1
source activate snn_bench

cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Start Time:    $(date)"
echo "=============================================="

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Starting ANN-to-SNN conversion benchmark..."
echo ""

python train_ann2snn.py \
    --device cuda:0 \
    --ann_epochs 10 \
    --batch_size 128 \
    --lr 1e-3 \
    --T_values 8 16 32 64 128 256 512 1024 \
    --num_runs 3 \
    --data_dir ./data \
    --results_dir ./results

echo ""
echo "ANN-to-SNN benchmark complete at $(date)"
echo ""

# Generate final comprehensive plots
python plot_final.py \
    --m1_dir $WORK/hpml/project/m1/results \
    --m2_dir $WORK/hpml/project/m2/results \
    --m3_dir $WORK/hpml/project/m3/results \
    --m4_dir ./results \
    --plots_dir ./plots

echo "All plots saved to ./plots/"
echo "Job finished at $(date)"
