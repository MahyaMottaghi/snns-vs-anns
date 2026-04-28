#!/bin/bash
#SBATCH -J final_plots
#SBATCH -o final_plots.o%j
#SBATCH -e final_plots.e%j
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00

module load anaconda3/2024.10-1
source activate snn_bench

cd $SLURM_SUBMIT_DIR

echo "Generating comprehensive comparison plots..."
echo "  M1: /work/ofx323/hpml/project/m1/results"
echo "  M2: /work/ofx323/hpml/project/m2/results"
echo "  M3: /work/ofx323/hpml/project/m3/results"
echo "  M4: ./results"

python plot_final.py \
    --m1_dir /work/ofx323/hpml/project/m1/results \
    --m2_dir /work/ofx323/hpml/project/m2/results \
    --m3_dir /work/ofx323/hpml/project/m3/results \
    --m4_dir ./results \
    --plots_dir ./plots

echo "Done at $(date)"
