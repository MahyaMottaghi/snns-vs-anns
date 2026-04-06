#  Spiking Neural Networks vs. Artificial Neural Networks — Training Efficiency, Accuracy, and Deployment


## I. SNN Neuron Models & Training Algorithms

### I-A. Surrogate Gradient Methods & BPTT

**[1] Neftci, E. O., Mostafa, H., & Zenke, F. (2019). "Surrogate gradient learning in spiking neural networks." *IEEE Signal Processing Magazine*, 36(6), 51–63.** [Primary]
Establishes the theoretical foundation for replacing the non-differentiable Heaviside spike function with smooth surrogates during backpropagation. Compares arctangent, sigmoid, and piecewise linear surrogates, showing arctangent and fast sigmoid achieve the best accuracy–stability tradeoff. Directly justifies the arctangent surrogate used in both Methods 1 and 2 of our study.

**[2] Eshraghian, J. K., et al. (2023). "Training spiking neural networks using lessons from deep learning." *Proceedings of the IEEE*, 111(9), 1016–1054.** [Primary]
Comprehensive tutorial bridging deep learning practices (batch normalization, dropout, learning rate scheduling) with SNN training. Introduces the SNNTorch library and demonstrates that standard deep learning pipelines adapt directly to SNNs with surrogate gradients. Primary reference for Method 1; provides the LIF convolutional architecture template used throughout our experiments.

**[3] Wu, Y., Deng, L., Li, G., Zhu, J., & Shi, L. (2018). "Spatio-temporal backpropagation for training high-performance spiking neural networks." *Frontiers in Neuroscience*, 12, 331.** [Primary]
Proposes STBP, computing gradients along both spatial (layer) and temporal (timestep) axes simultaneously. Demonstrates that combining spatial and temporal gradient pathways improves convergence over purely temporal BPTT. Provides the theoretical basis for how both SNNTorch and SpikingJelly handle gradient flow across the T-dimensional computation graph.

**[4] Zenke, F. & Ganguli, S. (2018). "SuperSpike: Supervised learning in multilayer spiking neural networks." *Neural Computation*, 30(6), 1514–1541.** [Secondary]
Introduces a voltage-based surrogate gradient for multi-layer SNN training on temporal classification tasks. Provides important context for understanding the gradient approximation tradeoffs that underpin all surrogate-based SNN training, and why different surrogate choices yield different convergence behaviors.

**[5] Werbos, P. J. (1990). "Backpropagation through time: What it does and how to do it." *Proceedings of the IEEE*, 78(10), 1550–1560.** [Secondary]
Foundational paper on BPTT for recurrent networks. The same unrolling technique applies to SNNs, where membrane state creates temporal dependencies analogous to RNN hidden states. Essential background for the parallel time complexity analysis in our study.

### I-B. Memory-Efficient & Online Training Alternatives

**[6] Bellec, G., et al. (2020). "A solution to the learning dilemma for recurrent networks of spiking neurons." *Nature Communications*, 11, 3625.** [Secondary]
Proposes e-prop, an eligibility-trace-based rule approximating BPTT with O(1) memory per timestep. While our study uses standard BPTT, e-prop represents the most promising direction for reducing the temporal memory overhead that limits SpikingJelly's backward pass at high T.

**[7] Xiao, M., Meng, Q., Zhang, Z., He, D., & Lin, Z. (2022). "Online training through time for spiking neural networks." *NeurIPS*, 35, 20717–20730.** [Secondary]
Proposes OTTT with constant memory consumption independent of T, addressing a key limitation identified in our SpikingJelly backward pass analysis where memory scales as O(T × batch × features). Demonstrates that online alternatives can eliminate the memory–timestep tradeoff.

---

## II. SNN Software Frameworks & GPU Acceleration

### II-A. PyTorch-Based Frameworks

**[8] Fang, W., et al. (2023). "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence." *Science Advances*, 9(40), eadi1480.** [Primary]
The SpikingJelly framework paper. Introduces multi-step simulation mode and CUDA-accelerated neuron backends (CuPy/Triton). Reports up to 11× training speedup over SNNTorch on Spiking ResNet-18 at T=32 using CuPy backend. Our study replicates this comparison using the torch backend, finding more moderate 1.2–1.7× speedups and identifying the backward pass as the limiting factor when custom kernels are not used.

**[9] Pehle, C. & Pedersen, J. E. (2021). "Norse — A library for gradient-based learning in spiking neural networks." *Zenodo*.** [Secondary]
Norse uses a functional programming design for SNN construction in PyTorch. Benchmarked alongside SNNTorch and SpikingJelly in Fang et al. (2023), showing intermediate performance. Provides context for the broader landscape of competing SNN frameworks.

**[10] Hazan, H., et al. (2018). "BindsNET: A machine learning-oriented spiking neural networks library in Python." *Frontiers in Neuroinformatics*, 12, 89.** [Secondary]
Early PyTorch-based SNN framework focused on biologically plausible STDP learning. Lacks surrogate gradient support but demonstrates the benefits of building SNN tools on GPU-accelerated deep learning frameworks. Historical context for SNNTorch and SpikingJelly evolution.

### II-B. Alternative Backends

**[11] Lenz, G. & Eshraghian, J. K. (2024). "Spyx: A library for just-in-time compiled optimization of spiking neural networks." *arXiv:2402.18994*.** [Secondary]
JAX-based SNN framework using XLA JIT compilation to match SpikingJelly CuPy performance without custom CUDA code. Benchmarks on N-MNIST show Spyx within 5% of SpikingJelly-CuPy speed. Demonstrates that compiler optimization can substitute for hand-written CUDA kernels, contextualizing our torch-backend results.

**[12] Nowotny, T., Turner, J. P., & Knight, J. C. (2022). "Loss shaping enhances exact gradient learning with EventProp in spiking neural networks." *arXiv:2212.01232*.** [Secondary]
Proposes mlGeNN framework compiling SNN simulations directly to CUDA from C++. Represents the extreme end of GPU optimization and provides context for why SpikingJelly's torch backend achieves moderate speedups versus fully compiled approaches.

---

## III. ANN-to-SNN Conversion Methods

**[13] Rueckauer, B., et al. (2017). "Conversion of continuous-valued deep networks to efficient event-driven networks for image classification." *Frontiers in Neuroscience*, 11, 682.** [Primary]
Foundational paper on ANN-to-SNN conversion via threshold balancing. Demonstrates that ReLU activations map to integrate-and-fire firing rates when thresholds are calibrated using activation statistics. Reports near-lossless accuracy but requires T ≥ 500 for rate convergence. Directly motivates Method 4 and the high T values (up to 1024) in our conversion benchmark.

**[14] Diehl, P. U., et al. (2015). "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing." *IJCNN*, 1–8.** [Primary]
Proposes data-driven threshold normalization where layer thresholds are set to maximum observed activation over calibration data. This "max" mode is the default in SpikingJelly's `ann2snn` converter used in our Method 4.

**[15] Bu, T., et al. (2023). "Optimal ANN-SNN conversion for high-performance and energy-efficient spiking neural networks." *ICLR*.** [Secondary]
Proposes optimized conversion minimizing timestep requirements through calibrated initial membrane potentials and quantization-aware training. Achieves competitive accuracy at T=32 rather than T ≥ 500, bridging the gap between conversion and direct training — relevant for interpreting our Method 4 results at low T values.

**[16] Wang, Z., et al. (2025). "Deep spiking neural networks with high representation similarity achieve high task-specific performance." *Nature Communications*, 16, 1070.** [Secondary]
Shows converted SNNs can match ANN accuracy with fewer timesteps by optimizing representation similarity between layers. Context for future optimization of the conversion approach to reduce the high T requirement found in our experiments.

---

## IV. SNN vs. ANN Benchmarking & Energy Analysis

**[17] NeuroBench Collaborative. (2025). "The NeuroBench framework for benchmarking neuromorphic computing algorithms and systems." *Nature Communications*, 16, 1840.** [Primary]
Establishes standardized metrics: correctness (accuracy, mAP), complexity (synaptic operations/inference, memory footprint, connection sparsity), and system-level (execution time, energy). Our study aligns with NeuroBench by reporting training time, forward/backward decomposition, and inference time as functions of T.

**[18] Dampfhoffer, M., et al. (2022). "Are SNNs really more energy-efficient than ANNs? An in-depth hardware-aware study." *IEEE TETCI*.** [Primary]
Rigorous analysis comparing SNN accumulate (AC) operations versus ANN multiply-and-accumulate (MAC) operations. Shows SNNs are only more energy-efficient when spike sparsity is very high (0.15–1.38 spikes per synapse per inference). Critically important context for our SNN vs. ANN comparison: energy efficiency claims require hardware-aware analysis, not just algorithmic comparison.

**[19] Open Neuromorphic Community. (2024). "Spiking Neural Network (SNN) Library Benchmarks." *open-neuromorphic.org/blog*.** [Primary]
Community benchmark comparing training speed of SNNTorch, SpikingJelly (torch/CuPy), Norse, Sinabs, Lava-DL, and Rockpool on LIF forward+backward passes. Reports SpikingJelly-CuPy at 0.26s combined. Our study extends this microbenchmark to full end-to-end training with accuracy measurement and timing decomposition.

**[20] A Practical Tutorial on Spiking Neural Networks (2025). *Preprints.org*, 202509.2072.** [Primary]
Benchmarks FCN on MNIST and VGG7 on CIFAR-10 across Lava, SpikingJelly, Norse, and PyTorch with multiple neuron models and encodings. Reports accuracy–energy tradeoffs showing SNNs can achieve up to 3× energy efficiency versus matched ANNs. Closest methodological precedent to our study, though without parallel computing timing decomposition.

**[21] Shen, J., et al. (2024). "Are conventional SNNs really efficient? A perspective from network quantization." *CVPR*.** [Secondary]
Challenges SNN efficiency claims by showing that quantized ANNs at equivalent bit budgets match SNN performance. Introduces the "bit budget" framework unifying SNNs and quantized ANNs. Important counterpoint to energy efficiency arguments: SNNs may not be inherently more efficient than properly quantized ANNs.

**[22] Gebregiorgis, A., et al. (2025). "Spike-based neuromorphic computing." *Microprocessors and Microsystems*, 105240.** [Secondary]
Comprehensive tutorial covering neuron models, learning algorithms, hardware architectures, and emerging materials. Provides broad context for positioning SNN software frameworks within the full neuromorphic computing stack.

---

## V. Neuromorphic Hardware & Deployment Platforms

**[23] Kudithipudi, D., Pandit, T., et al. (2025). "Neuromorphic computing at scale." *Nature*, 637, 801–812.** [Primary]
Multi-institutional review (23 authors) charting the roadmap for scalable neuromorphic systems. Calls for open ecosystems and standardized benchmarks. Led by the UTSA MATRIX AI Consortium, which also leads the THOR Neuromorphic Commons — the platform where our future hardware deployment will occur using the SpiNNaker2 system installed at UTSA.

**[24] Mayr, C., Hoeppner, S., & Furber, S. (2024). "SpiNNaker2: A large-scale neuromorphic system for event-based and asynchronous machine learning." *arXiv:2401.04491*.** [Primary]
Describes the SpiNNaker2 architecture: 153 ARM Cortex-M4F cores per chip with MAC and neuromorphic accelerators, 22nm FDSOI. Supports both SNNs and DNNs. Deployed at UTSA as part of THOR, making it the concrete future deployment target for models trained in this study — bridging the GPU training benchmarked here with actual neuromorphic hardware execution.

**[25] Davies, M., et al. (2018). "Loihi: A neuromorphic manycore processor with on-chip learning." *IEEE Micro*, 38(1), 82–99.** [Secondary]
Intel's Loihi processor with 128 neuromorphic cores and on-chip STDP. Relevant as an alternative deployment target for SpikingJelly models. Loihi and SpiNNaker2 represent the two main families of digital neuromorphic hardware currently accessible to researchers.

**[26] Orchard, G., et al. (2015). "Converting static image datasets to spiking neuromorphic datasets using saccades." *Frontiers in Neuroscience*, 9, 437.** [Secondary]
Describes creation of neuromorphic event-camera datasets (N-MNIST, N-Caltech101). Provides context for extending our MNIST benchmark to event-driven datasets where SNNs have a natural temporal advantage over ANNs.

---

## VI. Foundational Deep Learning References

**[27] He, K., et al. (2016). "Deep residual learning for image recognition." *CVPR*, 770–778.** [Secondary]
Introduces ResNet, whose spiking variants are the standard benchmark architecture for SNN framework comparisons. Fang et al. (2023) used Spiking ResNet-18 for their 11× speedup claim. Our smaller convolutional architecture provides complementary small-scale results.

**[28] Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance deep learning library." *NeurIPS*, 32.** [Primary]
The deep learning framework upon which SNNTorch, SpikingJelly, and our ANN baseline are built. PyTorch's autograd, CUDA dispatch, and `nn.Module` abstraction directly determine the overhead characteristics analyzed in our study. The ANN baseline (Method 3) uses pure PyTorch without any SNN extensions.

**[29] Pedersen, J. E., et al. (2024). "Neuromorphic Intermediate Representation (NIR)." *Nature Communications*, 15, 1.** [Secondary]
Defines a graph-based intermediate representation for SNN computational graphs enabling cross-framework interoperability. Relevant as a future tool for validating architectural equivalence between SNNTorch, SpikingJelly, and converted models.

**[30] Fang, W., et al. (2021). "Deep residual learning in spiking neural networks." *NeurIPS*, 34, 21056–21069.** [Secondary]
Demonstrates that residual connections can be adapted for SNNs using membrane potential shortcuts. Achieves state-of-the-art on CIFAR-10 and DVS-CIFAR10 with directly trained Spiking ResNets. Relevant for understanding scalability of the frameworks benchmarked in our study to larger architectures.

---
