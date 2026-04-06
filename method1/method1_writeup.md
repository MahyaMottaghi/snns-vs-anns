# Method 1: Direct SNN Training via Surrogate Gradients (SNNTorch)

## Problem Statement

Spiking Neural Networks (SNNs) process information through discrete binary spikes rather than continuous activations, making them inherently event-driven and energy-efficient. However, the non-differentiable nature of the spike threshold (a Heaviside step function) prevents standard backpropagation from being applied directly. This project benchmarks surrogate gradient training of SNNs using the SNNTorch framework, measuring training time, accuracy, and computational overhead as the number of simulation timesteps T is varied from 4 to 64.

## Hardware and Software Configuration

All experiments were conducted on the UTSA ARC high-performance computing cluster, submitted via the SLURM workload manager to the `gpu1v100` partition. The hardware and software configuration is as follows:

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
| SNN Framework | SNNTorch (pip install) |
| Python | 3.10 (Anaconda 2024.10-1) |
| Job Scheduler | SLURM (sbatch submission, single-node single-GPU jobs) |
| Max Job Duration | 72 hours (gpu1v100 partition limit) |

## Chosen Algorithm and Method

SNNTorch is a PyTorch-based library that implements Leaky Integrate-and-Fire (LIF) neurons as standard `nn.Module` layers. During the forward pass, the true binary spike is produced whenever a neuron's membrane potential exceeds its threshold. During the backward pass, an arctangent surrogate gradient replaces the undefined Heaviside derivative, enabling Backpropagation Through Time (BPTT) across T timesteps. The network architecture is a convolutional SNN consisting of two convolutional layers (12 and 32 filters of size 5×5), each followed by a LIF neuron layer and max-pooling, then a fully connected layer with 800 LIF neurons, and a final linear output layer. The model contains 428,354 trainable parameters and is trained on the MNIST dataset (60,000 training / 10,000 test images, 28×28 grayscale) using the Adam optimizer (lr = 1e-3), cross-entropy loss computed on output membrane potentials at each timestep, batch size 128, and LIF decay rate β = 0.95.

## Underlying Communication Pattern

The forward pass iterates over T timesteps sequentially in a Python for-loop. At each timestep, the same input image is presented, and the following operations execute on the GPU: (1) convolution / linear matrix multiplication, (2) LIF membrane potential update via element-wise addition and decay, and (3) spike generation via threshold comparison. Each iteration launches separate CUDA kernels for these operations. The membrane state from timestep t is consumed by timestep t+1, creating a strict sequential dependency along the temporal axis. Within each timestep, all samples in the batch are processed in parallel via standard PyTorch data parallelism on a single NVIDIA Tesla V100S GPU.

## Data Structures, Datasets, and Hyperparameters

The primary data structures are 4D tensors for input images (batch × channels × height × width) and per-layer membrane potential state tensors that persist across timesteps. The membrane potential and spike outputs are accumulated into lists of length T before being stacked for loss computation. The dataset is MNIST with normalization to zero mean and unit standard deviation. Hyperparameters: T ∈ {4, 8, 16, 32, 64}, batch size = 128, learning rate = 1e-3 (Adam), β = 0.95 (LIF decay), arctangent surrogate slope = default, 10 epochs per T value, 3 runs per T value for variance estimation.

## Synchronization Overheads and Contention

The dominant overhead is the Python-level for-loop over T timesteps. Each loop iteration incurs: (a) Python interpreter overhead returning control between CUDA kernel launches, (b) CUDA kernel launch latency (~5–10 μs per kernel), and (c) implicit GPU synchronization at each LIF neuron state update since the next timestep's computation depends on the current membrane potential. With approximately 8 GPU kernel launches per timestep (conv, pool, LIF×4 layers), the total kernel launch overhead scales as O(8T) per batch. At T=64, this amounts to ~512 kernel launches per batch, multiplied by 468 batches per epoch. There is no inter-GPU communication (single GPU), and data loading is overlapped using 2 worker threads with pinned memory.

## Parallel Time Complexity

For a single training iteration (one batch), the computation time is O(T × C_layer), where C_layer is the per-timestep cost of the CNN forward pass. The communication/synchronization time is O(T × K), where K is the per-timestep kernel launch and Python loop overhead. The backward pass performs BPTT, which unrolls the computation graph across T timesteps, yielding backward time approximately 0.5–1.0× the forward time. Total epoch time = (T × C_layer + T × K) × num_batches + data_loading_time.

## Timing and Experiment Details

All experiments were run on the UTSA ARC cluster using a single NVIDIA Tesla V100S GPU (gpu1v100 partition). Five T values were swept: 4, 8, 16, 32, 64. Each configuration was run for 10 epochs and repeated 3 times. Per-epoch wall-clock time was measured, with the forward pass and backward pass timed separately using `torch.cuda.Event` for GPU-precise timing. A warmup batch was executed before timing to eliminate cold-start artifacts.

## Performance Results

**Training Time (Figure: snntorch_time_vs_T.png):** Epoch time scales nearly linearly with T: 14.5s at T=4, 19.7s at T=8, 31.9s at T=16, 56.3s at T=32, and 105.1s at T=64. The approximately 7.2× increase from T=4 to T=64 (vs. the theoretical 16× if perfectly linear) indicates that fixed overheads (data loading, optimizer step) amortize the per-timestep cost at low T values.

**Time Breakdown (Figure: snntorch_time_breakdown.png):** The forward pass dominates at all T values, accounting for 54% of epoch time at T=4 and 66% at T=64. The backward pass accounts for 33% (T=4) to 33% (T=64) of total time. Data loading is a relatively small fraction (1–14%) and varies with system I/O load.

**Accuracy (Figure: snntorch_accuracy_vs_T.png):** Accuracy improves with T: 89.9% at T=4, 92.1% at T=8, 96.2% at T=16, 98.0% at T=32, then slightly decreases to 96.3% at T=64. The accuracy at T=4 shows high variance (σ ≈ 2.4%) due to insufficient temporal information for the LIF neurons. Peak accuracy is at T=32 (98.0%), after which additional timesteps cause overfitting to temporal noise.

**Loss Curves (Figure: snntorch_loss_curves.png):** All T values converge within 10 epochs. Higher T values start with larger initial loss (because loss is summed across T timesteps) but converge to similar relative values. The T=4 and T=8 configurations converge fastest in absolute terms.

## Conclusions

SNNTorch provides a straightforward, PyTorch-native approach to SNN training. However, the Python-level timestep loop creates a significant computational bottleneck that scales linearly with T. The forward pass consistently dominates training time because each timestep launches multiple sequential CUDA kernels. The sweet spot for this architecture on MNIST is T=32, which achieves 98.0% accuracy in 56.3 seconds per epoch. Beyond T=32, accuracy degrades while training time nearly doubles, indicating diminishing returns. The key limitation is the per-timestep kernel launch overhead inherent to the Python loop — this is precisely the overhead that SpikingJelly (Method 2) addresses through multi-step CUDA kernel fusion. Future improvements could include implementing custom CUDA kernels to fuse the timestep loop, using mixed-precision (FP16) training to reduce memory bandwidth, or deploying the trained model on neuromorphic hardware (e.g., SpiNNaker2 via the UTSA THOR platform) for real energy measurements.
