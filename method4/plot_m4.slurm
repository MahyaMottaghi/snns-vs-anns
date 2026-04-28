#!/bin/bash
#SBATCH -J m4_plot
#SBATCH -o m4_plot.o%j
#SBATCH -e m4_plot.e%j
#SBATCH -p compute1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00

module load anaconda3/2024.10-1
source activate snn_bench

cd $SLURM_SUBMIT_DIR

python ann2snn_plot.py --results_dir ./results --plots_dir ./plots

echo "M4 plots saved to ./plots/"
