#!/bin/bash
# =============================================================================
# UTSA ARC - ONE-TIME SETUP: Create conda environment for SNN experiments
# =============================================================================
# Run this ONCE from a GPU node interactively:
#
#   ssh -p 22 abc123@arc.utsa.edu
#   srun -p gpu1v100 -N 1 -n 1 -t 01:00:00 --pty bash
#   bash setup_env.sh
#
# Replace abc123 with your actual UTSA ID.
# =============================================================================

set -e  # Exit on any error

echo "================================================"
echo "  Setting up SNN conda environment on UTSA ARC"
echo "================================================"

# Load Anaconda module (only available on job nodes, NOT login nodes)
module load anaconda3/2024.10-1

# Create conda environment
ENV_NAME="snn_bench"
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate it
source activate $ENV_NAME

# Install PyTorch with CUDA support (V100S supports CUDA 11.x and 12.x)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SNN frameworks
echo "Installing SNNTorch..."
pip install snntorch

echo "Installing SpikingJelly (for Method 2)..."
pip install spikingjelly

# Install utilities
echo "Installing plotting/data utilities..."
pip install matplotlib pandas

# Verify GPU access
echo ""
echo "================================================"
echo "  Verifying installation..."
echo "================================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device:      {torch.cuda.get_device_name(0)}')
    print(f'CUDA version:    {torch.version.cuda}')

import snntorch
print(f'SNNTorch OK')

import spikingjelly
print(f'SpikingJelly OK')

print()
print('All packages installed successfully!')
"

echo ""
echo "================================================"
echo "  Setup complete!"
echo "  To use this environment in future sessions:"
echo "    module load anaconda3"
echo "    source activate $ENV_NAME"
echo "================================================"
