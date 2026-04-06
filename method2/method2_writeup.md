# Method 2: Event-Driven SNN Computation (SpikingJelly)

## Problem Statement

While surrogate gradient training enables effective SNN optimization, the per-timestep Python loop used by frameworks like SNNTorch introduces substantial kernel launch and interpreter overhead that scales linearly with T. This project benchmarks SpikingJelly's multi-step execution mode, which fuses the temporal simulation into fewer CUDA operations, and directly compares its training time and accuracy against SNNTorch (Method 1) on the identical architecture, dataset, and hyperparameters.

## Hardware and Software Configuration

All experiments were conducted on the same UTSA ARC cluster node type as Method 1 to ensure a fair timing comparison. Jobs were submitted via SLURM to the `gpu1v100` partition.

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla V100S-PCIE-32GB (1× per node) |
| GPU Memory | 32 GB HBM2, 1,134 GB/s bandwidth |
| GPU Cores | 5,120 CUDA cores, 640 Tensor cores |
| CPU | Intel Xeon CascadeLake (40 physical cores, 80 hyperthreads) |
| System RAM | Up to 192 GB DDR4 |
| Local Scratch | 1.5 TB per node |
| PyTorch | 2.5.1+cu121 |
| CUDA Toolkit | 12.1 |
| SNN Framework | SpikingJelly (torch backend, pip install) |
| Python | 3.10 (Anaconda 2024.10-1) |
| Job Scheduler | SLURM (sbatch submission, single-node single-GPU jobs) |
| Max Job Duration | 72 hours (gpu1v100 partition limit) |

Note: SpikingJelly was configured with the `torch` backend for neuron computation. The optional `cupy` and `triton` backends, which fuse the entire LIF forward and backward pass into single custom CUDA kernels, were not used in this experiment in order to isolate the contribution of multi-step tensor batching alone.

## Chosen Algorithm and Method

SpikingJelly is a PyTorch-based SNN framework that provides a multi-step simulation mode (`step_mode='m'`). In multi-step mode, all T timesteps are processed in a single call to each layer rather than iterating in Python. The framework reshapes the input to (T, batch, C, H, W) and passes the full temporal sequence through each layer at once, allowing internal CUDA optimizations and reducing Python-level overhead. The algorithm is identical to Method 1 — surrogate gradient BPTT with arctangent surrogate — but the execution strategy differs fundamentally. The architecture is matched exactly: Conv2d(1,12,5) → LIF → MaxPool → Conv2d(12,32,5) → LIF → MaxPool → Flatten → Linear(512,800) → LIF → Linear(800,10). The final output layer produces continuous membrane potentials (no LIF neuron), and cross-entropy loss is computed per-timestep and summed, identical to Method 1. The model has 428,354 parameters. The LIF time constant τ = 20.0, which is equivalent to SNNTorch's β = 0.95 via the relationship β = 1 − 1/τ.

## Underlying Communication Pattern

The critical architectural difference is the elimination of the Python for-loop. In Method 1, each timestep launches separate CUDA kernels sequentially. In SpikingJelly's multi-step mode, the convolution and linear layers receive the full (T, batch, ...) tensor and process all timesteps in a single batched operation — effectively treating the temporal dimension as an extended batch dimension for non-stateful layers. For LIF neuron layers, the sequential temporal dependency (membrane state at t depends on t−1) still exists, but SpikingJelly handles this internally in optimized C++/CUDA code rather than through Python iteration. When using the torch backend (as in this experiment), the neuron dynamics are computed via vectorized PyTorch operations. With the optional CuPy or Triton backends, the entire T-step LIF forward and backward pass can be fused into a single custom CUDA kernel. The batch dimension remains parallel across all samples, and all computation occurs on a single NVIDIA Tesla V100S GPU.

## Data Structures, Datasets, and Hyperparameters

Input tensors are reshaped to 5D: (T, batch, channels, height, width), where the temporal dimension is prepended. Each LIF neuron layer maintains internal membrane potential state that is reset between batches via `functional.reset_net()`. The dataset is MNIST with identical preprocessing to Method 1 (normalization to zero mean, unit std). Hyperparameters are matched exactly: T ∈ {4, 8, 16, 32, 64}, batch size = 128, learning rate = 1e-3 (Adam), τ = 20.0 (equivalent to β = 0.95), arctangent surrogate, 10 epochs, 3 runs per T. The torch backend was used (no CuPy/Triton compilation required).

## Synchronization Overheads and Contention

By eliminating the Python timestep loop, SpikingJelly removes the O(T) Python interpreter overhead and reduces the number of CUDA kernel launches. For non-stateful layers (Conv2d, Linear, MaxPool), the T timesteps are batched into a single kernel call, reducing launches from O(T × L) to O(L), where L is the number of layers. For LIF neuron layers, the temporal dependency still requires sequential processing, but this occurs within compiled PyTorch operations rather than Python-level iteration. The torch backend achieves this through vectorized tensor operations; the CuPy backend would further fuse the entire LIF dynamics into a single kernel. The remaining overheads are: (a) the memory cost of materializing the full (T, batch, C, H, W) tensor, which grows linearly with T, and (b) the backward pass through BPTT, which must still unroll the computation graph across T timesteps. Data loading uses 2 worker threads with pinned memory, identical to Method 1.

## Parallel Time Complexity

For a single batch, computation time is O(T × C_layer), identical to Method 1 in raw FLOPs. However, the communication/synchronization time is reduced from O(T × K) to O(K_fused), where K_fused ≪ T × K because kernel launches are batched. The backward pass time is O(T × B_layer), where BPTT unrolls across T steps. In practice, the backward pass showed less speedup than the forward pass because BPTT inherently requires sequential gradient propagation through time. Total epoch time = (T × C_layer + K_fused) × num_batches + data_loading_time.

## Timing and Experiment Details

Experiments were conducted on the same UTSA ARC gpu1v100 partition (single NVIDIA Tesla V100S) as Method 1. The torch backend was used for SpikingJelly neuron computation. Five T values were swept: 4, 8, 16, 32, 64. Each was run for 10 epochs with 3 repetitions. Timing used `torch.cuda.Event` for GPU-precise measurement with a warmup batch to eliminate cold-start effects. Forward and backward passes were timed separately.

## Performance Results

**Training Time (Figure: spikingjelly_time_vs_T.png):** Epoch time increases with T: 12.2s at T=4, 14.5s at T=8, 19.2s at T=16, 36.1s at T=32, and 79.5s at T=64. Compared to SNNTorch, SpikingJelly is faster at every T value. The speedup is 1.19× at T=4, 1.36× at T=8, 1.66× at T=16, 1.56× at T=32, and 1.32× at T=64.

**Time Breakdown (Figure: spikingjelly_time_breakdown.png):** The forward pass is dramatically faster than SNNTorch — 2.3s vs. 7.8s at T=4 (3.4× faster) and 24.6s vs. 69.4s at T=64 (2.8× faster). This confirms that multi-step mode successfully reduces kernel launch overhead for the forward pass. However, the backward pass is slower than SNNTorch at high T: 53.0s vs. 35.2s at T=64. This is because SpikingJelly's multi-step backward pass materializes the full computation graph for all T steps simultaneously, consuming more memory and increasing BPTT overhead. Data loading time is higher at low T (6.9s at T=4), likely because the short computation time makes CPU data preparation the bottleneck.

**Accuracy (Figure: spikingjelly_accuracy_vs_T.png):** Accuracy is high and stable across all T values: 97.4% at T=4, 97.5% at T=8, 97.9% at T=16, 97.8% at T=32, and 97.5% at T=64. Notably, SpikingJelly achieves 97.4% at T=4 versus SNNTorch's 89.9%, a 7.5 percentage point advantage. This is because SpikingJelly's output layer produces continuous membrane potentials (no final LIF neuron), providing richer gradient signal to the loss function even at very low T. The accuracy plateau around 97.5–98% with minimal variance (σ < 0.3%) demonstrates that the multi-step execution produces equivalent learning dynamics to the Python loop approach.

**Speedup Analysis:** The speedup over SNNTorch is most pronounced in the forward pass (2.8–3.4×) but partially offset by slower backward pass performance at high T. The net speedup peaks at T=16 (1.66×) where the forward pass savings dominate and the backward pass penalty is moderate. At T=64, the backward pass becomes the bottleneck (53s vs. 35s), reducing net speedup to 1.32×.

## Conclusions

SpikingJelly's multi-step mode delivers a consistent 1.2–1.7× training speedup over SNNTorch across all T values by eliminating the Python-level timestep loop in the forward pass. The forward pass alone is 2.8–3.4× faster, confirming that CUDA kernel launch overhead is the primary bottleneck in SNNTorch's approach. However, the backward pass with the torch backend does not achieve the same speedup — it is actually slower at T≥32 due to the memory overhead of materializing the full temporal computation graph. This means the overall speedup is moderate rather than the 11× reported by Fang et al. (2023), who used the CuPy backend with custom CUDA kernels on a Spiking ResNet-18. The torch backend used in this experiment does not fuse the LIF backward pass into a single kernel, leaving room for further optimization. Accuracy is equivalent or better than SNNTorch at all T values, with the advantage being most striking at low T (97.4% vs. 89.9% at T=4) due to the continuous output layer design. The key limitation is memory consumption: multi-step mode materializes tensors of shape (T, batch, C, H, W), which scales linearly with T and may limit applicability to larger models or higher T values. Future work includes benchmarking with the CuPy backend for full kernel fusion, profiling GPU memory usage as a function of T, and deploying trained models on the SpiNNaker2 neuromorphic hardware available through the UTSA THOR platform.
