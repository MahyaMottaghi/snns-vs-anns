# Parallel Computing Analysis of Spiking vs. Artificial Neural Networks: Training Efficiency, Accuracy, and the Cost of Temporal Computation

## 1. Problem Statement

Spiking Neural Networks (SNNs) have emerged as an energy-efficient alternative to conventional Artificial Neural Networks (ANNs), encoding information through discrete binary spikes that enable event-driven computation on neuromorphic hardware. However, this biological plausibility comes at a computational cost: SNN training on GPU hardware requires simulating T discrete timesteps per input, introducing sequential overhead absent in standard ANN training. The central question of this study is: **what is the actual computational cost of "going spiking" on GPU hardware, and how do different SNN training paradigms compare against ANN baselines in terms of training time, inference latency, and accuracy?**

This project benchmarks four methods spanning the SNN–ANN spectrum on the MNIST image classification task using a matched convolutional architecture with 428,354 parameters:

| Method | Approach | Training | Inference |
|--------|----------|----------|-----------|
| M1: SNNTorch | Direct SNN training (surrogate gradient BPTT, Python timestep loop) | SNN | SNN |
| M2: SpikingJelly | Direct SNN training (surrogate gradient BPTT, fused multi-step CUDA) | SNN | SNN |
| M3: ANN Baseline | Standard CNN training (ReLU, backpropagation) | ANN | ANN |
| M4: ANN-to-SNN | Train ANN first, then convert to SNN via threshold balancing | ANN | SNN |

Methods 1 and 2 represent two GPU execution strategies for the same SNN algorithm, isolating the impact of CUDA kernel fusion on training speed. Method 3 provides the performance ceiling (accuracy) and timing floor (speed) against which SNN overhead is measured. Method 4 represents the conversion paradigm, which achieves near-ANN accuracy but trades latency for fidelity by requiring high timestep counts (T ≥ 256) during inference. Together, these four methods map out the accuracy–latency–training-cost tradeoff space for spike-based computation on conventional GPU hardware.

## 2. Hardware and Software Configuration

All experiments were conducted on the UTSA ARC high-performance computing cluster, submitted via the SLURM workload manager to the `gpu1v100` partition.

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
| Python | 3.10 (Anaconda 2024.10-1) |
| SNN Frameworks | SNNTorch (M1), SpikingJelly torch backend (M2, M4) |
| Job Scheduler | SLURM (sbatch, single-node single-GPU jobs) |

## 3. Shared Architecture and Dataset

All four methods use the same convolutional backbone with matched depth and parameter count to ensure a fair comparison:

| Layer | SNN (M1, M2) | ANN (M3) | Conversion Source (M4) |
|-------|-------------|----------|----------------------|
| 1 | Conv2d(1,12,5) → LIF → MaxPool(2) | Conv2d(1,12,5) → ReLU → MaxPool(2) | Conv2d(1,12,5) → ReLU → AvgPool(2) |
| 2 | Conv2d(12,32,5) → LIF → MaxPool(2) | Conv2d(12,32,5) → ReLU → MaxPool(2) | Conv2d(12,32,5) → ReLU → AvgPool(2) |
| 3 | Linear(512,800) → LIF | Linear(512,800) → ReLU | Linear(512,800) → ReLU |
| 4 | Linear(800,10) | Linear(800,10) | Linear(800,10) |
| **Parameters** | **428,354** | **428,354** | **428,354** |

Note: M4 uses AvgPool instead of MaxPool because max-pooling is incompatible with rate-based ANN-to-SNN conversion.

**Dataset:** MNIST (60,000 train / 10,000 test, 28×28 grayscale), normalized to zero mean and unit standard deviation. No data augmentation.

**Shared hyperparameters:** Adam optimizer (lr = 1e-3), cross-entropy loss, batch size = 128, 10 epochs for training methods (M1, M2, M3), 3 runs per configuration.

---

## 4. Method 1: Direct SNN Training via SNNTorch

### 4.1 Algorithm and Method

SNNTorch implements Leaky Integrate-and-Fire (LIF) neurons as standard `nn.Module` layers. During the forward pass, the true binary spike is produced when membrane potential exceeds threshold. During the backward pass, an arctangent surrogate gradient replaces the undefined Heaviside derivative, enabling Backpropagation Through Time (BPTT) across T timesteps. The same input image is presented at each timestep (rate coding via repeated presentation). The LIF decay rate is β = 0.95.

### 4.2 Communication Pattern

The forward pass iterates over T timesteps in a Python for-loop. At each iteration, the interpreter returns control to Python between CUDA kernel launches. Each timestep launches approximately 8 GPU operations (conv, pool, LIF state update per layer), yielding ~8T kernel launches per batch. At T=64 with 468 batches/epoch, this amounts to ~240,000 kernel launches per epoch, each incurring ~5–10 μs of dispatch overhead. Membrane state at timestep t is consumed by t+1, creating a strict sequential dependency along the temporal axis. Within each timestep, all batch samples are processed in parallel via PyTorch data parallelism.

### 4.3 Synchronization Overheads

The dominant overhead is the Python-level for-loop: each iteration incurs (a) Python interpreter overhead, (b) CUDA kernel launch latency, and (c) implicit GPU synchronization at each LIF neuron state update. The cumulative overhead scales as O(T × K), where K ≈ 50–80 μs per timestep. No inter-GPU communication occurs (single GPU). Data loading is overlapped using 2 CPU worker threads with pinned memory.

### 4.4 Parallel Time Complexity

For a single training batch: computation = O(T × C_layer), communication/synchronization = O(T × K_launch), backward = O(T × B_layer). Total epoch time = (T × (C + K + B)) × N_batch + D, where N_batch = 468 and D = data loading time.

### 4.5 Experiment Details

T values swept: {4, 8, 16, 32, 64}. Each trained for 10 epochs, repeated 3 times (150 total runs). Timing via `torch.cuda.Event` with GPU warmup batch.

### 4.6 Performance Results

**Training Time:** Epoch time scales nearly linearly with T: 14.5s (T=4), 19.7s (T=8), 31.9s (T=16), 56.3s (T=32), 105.1s (T=64). The forward pass dominates, accounting for 54% at T=4 and 66% at T=64. Backward pass (BPTT) accounts for ~33%.

**Accuracy:** 89.9% (T=4), 92.1% (T=8), 96.2% (T=16), 98.0% (T=32), 96.3% (T=64). Peak at T=32. High variance at T=4 (σ ≈ 2.4%).

### 4.7 Conclusions for Method 1

The Python-level timestep loop creates O(T) kernel launch overhead. The sweet spot is T=32 (98.0%, 56.3s/epoch). Beyond T=32, accuracy degrades while time doubles.

---

## 5. Method 2: Event-Driven SNN Computation via SpikingJelly

### 5.1 Algorithm and Method

SpikingJelly uses the identical surrogate gradient BPTT algorithm as Method 1 but executes via multi-step mode (`step_mode='m'`). The full temporal sequence (T, batch, C, H, W) is passed through each layer in a single call. Non-stateful layers treat T as an extended batch dimension. LIF neuron layers handle sequential dependency internally through vectorized PyTorch operations (torch backend). τ = 20.0 (equivalent to β = 0.95).

Note: The `torch` backend was used. The optional `cupy` and `triton` backends were not used, isolating the contribution of multi-step tensor batching alone.

### 5.2 Communication Pattern

The Python timestep loop is eliminated. Non-stateful layers process the full (T, batch, ...) tensor in a single kernel call, reducing launches from O(T × L) to O(L). LIF layers still process T steps sequentially but within compiled PyTorch code, eliminating Python interpreter returns.

### 5.3 Synchronization Overheads

Kernel launch overhead drops from O(T) to O(1) for Conv2d/Linear layers. Remaining overhead: (a) LIF sequential updates within compiled code, (b) memory pressure from materializing the full (T, batch, C, H, W) tensor.

### 5.4 Parallel Time Complexity

Computation = O(T × C_layer) (same FLOPs as M1). Communication = O(K_fused) ≪ O(T × K_launch). Backward = O(T × B_layer) + memory overhead. The key difference: removal of the O(T × K_launch) term.

### 5.5 Performance Results

**Training Time and Speedup over Method 1:**

| T | SNNTorch (s) | SpikingJelly (s) | Net Speedup | Fwd Speedup | Bwd Speedup |
|---|-------------|-----------------|-------------|-------------|-------------|
| 4 | 14.5 | 12.2 | 1.19× | 3.42× | 1.55× |
| 8 | 19.7 | 14.5 | 1.36× | 2.88× | 1.28× |
| 16 | 31.9 | 19.2 | **1.66×** | 2.70× | 0.83× |
| 32 | 56.3 | 36.1 | 1.56× | 2.54× | 0.91× |
| 64 | 105.1 | 79.5 | 1.32× | 2.83× | 0.66× |

The forward pass is consistently 2.5–3.4× faster. The backward pass is *slower* at T ≥ 16 (0.66–0.91×) due to memory pressure, capping net speedup at 1.66×.

**Accuracy:** 97.4% (T=4), 97.5% (T=8), 97.9% (T=16), 97.8% (T=32), 97.5% (T=64). Higher and more stable than SNNTorch, especially at T=4 (97.4% vs. 89.9%).

**Inference:** Speedup grows with T: 1.23× (T=4) to 4.73× (T=64), because no backward pass penalty exists.

### 5.6 Conclusions for Method 2

Multi-step mode delivers 1.2–1.7× net training speedup and up to 4.73× inference speedup. The backward pass becomes the bottleneck at high T. Results are below the 11× reported by Fang et al. (2023) who used CuPy backend with custom CUDA kernels.

---

## 6. Method 3: ANN Baseline (PyTorch)

*(Results and detailed analysis to be added upon completion of experiments.)*

Method 3 trains the same architecture with ReLU instead of LIF, using standard single-pass backpropagation (no timestep loop, no BPTT). This serves as the accuracy ceiling and timing floor. The ANN has no T parameter — each image requires a single forward pass. Comparing M1/M2 training times against M3 directly quantifies the "cost of going spiking."

---

## 7. Method 4: ANN-to-SNN Conversion

*(Results and detailed analysis to be added upon completion of experiments.)*

Method 4 trains a ReLU ANN to convergence, then converts it to an IF-neuron SNN using SpikingJelly's `ann2snn` module with threshold balancing. Only inference is measured, at T ∈ {8, 16, 32, 64, 128, 256, 512, 1024}. This method upper-bounds accuracy (approaching the source ANN) while lower-bounding latency efficiency (requiring T ≥ 256 for rate convergence). Comparing M4 at matched T against M1/M2 illustrates why surrogate-gradient SNNs dominate low-latency applications while conversion dominates high-accuracy requirements.

---

## 8. Cross-Method Comparison

*(Full comparison to be completed with all four methods. Will include:)*

1. Accuracy vs. T — M1, M2, M4 curves + M3 horizontal baseline
2. Training time vs. T — M1, M2 curves + M3 horizontal baseline
3. Inference time vs. T — all four methods
4. Speedup — SpikingJelly over SNNTorch (forward, backward, net)
5. Time breakdown — forward / backward / data loading stacked bars
6. ANN-to-SNN convergence — M4 accuracy approaching M3 as T increases
7. Summary table — best accuracy, optimal T, epoch time, inference time, parameters

---

## 9. Preliminary Conclusions

1. **The Python timestep loop is a significant bottleneck:** Eliminating it yields 2.5–3.4× forward pass speedup.
2. **The backward pass is the new bottleneck:** SpikingJelly backward is slower at T ≥ 16 due to memory overhead, capping net speedup at 1.66×.
3. **Inference benefits more than training:** Up to 4.73× inference speedup since no backward pass penalty.
4. **Accuracy is equivalent when architectures match:** Both frameworks achieve ~97–98% on MNIST.
5. **The accuracy–latency sweet spot is T=16–32:** 97–98% accuracy with moderate overhead.

**Pending (M3, M4):** Quantifying exact SNN vs. ANN training overhead, minimum T for converted SNNs to match direct training, and whether conversion offers advantages at practical T values.

## References

[1] Neftci, E. O., et al. (2019). Surrogate gradient learning in spiking neural networks. *IEEE SPM*, 36(6), 51–63.

[2] Eshraghian, J. K., et al. (2023). Training spiking neural networks using lessons from deep learning. *Proc. IEEE*, 111(9), 1016–1054.

[3] Fang, W., et al. (2023). SpikingJelly: An open-source machine learning infrastructure platform. *Science Advances*, 9(40), eadi1480.

[4] Rueckauer, B., et al. (2017). Conversion of continuous-valued deep networks to efficient event-driven networks. *Frontiers in Neuroscience*, 11, 682.

[5] He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

[6] Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*, 32.

[7] Wu, Y., et al. (2018). Spatio-temporal backpropagation for training high-performance spiking neural networks. *Frontiers in Neuroscience*, 12, 331.

[8] NeuroBench Collaborative. (2025). The NeuroBench framework. *Nature Communications*, 16, 1840.

[9] Kudithipudi, D., et al. (2025). Neuromorphic computing at scale. *Nature*, 637, 801–812.

[10] Mayr, C., et al. (2024). SpiNNaker2. *arXiv:2401.04491*.
