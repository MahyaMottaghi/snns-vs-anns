
# Methods Description & Bibliography
**Spiking Neural Networks vs. Artificial Neural Networks — Training Efficiency, Accuracy, and Deployment**
  

---

## Methods

### 1. Direct SNN Training via Surrogate Gradients — SNNTorch

Spiking Neural Networks fire discrete binary spikes rather than continuous activations, making the standard backpropagation gradient undefined at the spike threshold. Surrogate gradient methods resolve this by substituting a smooth, differentiable approximation of the Heaviside step function during the backward pass — SNNTorch uses an arctangent surrogate by default — while keeping the true binary spike in the forward pass [[1]](https://doi.org/10.1109/JPROC.2023.3308088) [[2]](https://doi.org/10.1109/MSP.2019.2931595).

SNNTorch is a PyTorch-based library that exposes Leaky Integrate-and-Fire (LIF) neurons as standard `nn.Module` layers, allowing SNN architectures to be defined and trained with the same syntax as ordinary deep networks. Backpropagation Through Time (BPTT) is performed across a user-defined number of timesteps *T*, making *T* the primary latency–accuracy knob for this study: smaller *T* means faster inference but typically lower accuracy [[1]](https://doi.org/10.1109/JPROC.2023.3308088).

In this project, SNNTorch is used to train and evaluate matched SNN architectures on both dense datasets (MNIST, CIFAR-10) and sparse/event-driven datasets (N-MNIST, DVS-CIFAR10). 

---

### 2. Event-Driven SNN Computation — SpikingJelly

SpikingJelly is a full-stack, PyTorch-based SNN infrastructure platform that provides CUDA-accelerated single-step and multi-step neuron simulations, achieving up to 11× training speedup over naive Python loops. SpikingJelly is optimized for high-throughput research and ships with built-in preprocessing pipelines for neuromorphic datasets including DVS-CIFAR10 and DVS128 Gesture [[4]](https://doi.org/10.1126/sciadv.adi1480).

This framework handles the sparse/event-driven half of the experimental pipeline, where spatiotemporal event tensors must be binned into *T* discrete frames before network ingestion [[4]](https://doi.org/10.1126/sciadv.adi1480) [[5]](https://doi.org/10.3389/fnins.2015.00437).

---

### 3. ANN-to-SNN Conversion — Rate-Based Threshold Balancing

ANN-to-SNN conversion is an alternative to direct surrogate-gradient training: a standard ReLU network is first trained to convergence, then its weights are transferred to an equivalent network of integrate-and-fire neurons whose firing thresholds are calibrated so that each neuron's average firing rate matches the original ReLU activation magnitude. This approach can achieve near-lossless accuracy, but it typically requires a large number of timesteps (*T* ≥ 500) to allow firing rates to stabilize — the inverse of the low-latency goal [[6]](https://doi.org/10.3389/fnins.2017.00682).

In this study, ANN-to-SNN conversion serves as a reference point that upper-bounds achievable accuracy while lower-bounding latency efficiency. Comparing converted SNNs against directly trained ones at matched *T* values illustrates the accuracy–latency trade-off between the two paradigms and provides context for why surrogate-gradient SNNs dominate low-latency applications [[6]](https://doi.org/10.3389/fnins.2017.00682) [[7]](https://doi.org/10.1109/TNNLS.2023.3337176).

---

### 4. ANN Baseline — PyTorch / ResNet

The ANN baseline uses standard convolutional architectures (FCN for MNIST, ResNet-20 for CIFAR-10) implemented in PyTorch, trained with cross-entropy loss and the Adam optimizer. These serve as the performance ceiling and energy reference point against which SNN efficiency claims are evaluated. All ANN and SNN models are matched in depth and parameter count to ensure a fair architectural comparison [[8]](https://arxiv.org/abs/1512.03385) [[9]](https://arxiv.org/abs/1912.01703).


---

## Bibliography

| # | Citation |
|---|----------|
| [1] | Eshraghian, J. K., Ward, M., Neftci, E., Wang, X., et al. (2023). "Training Spiking Neural Networks Using Lessons from Deep Learning." *Proceedings of the IEEE*, 111(9), 1016–1054. → [doi:10.1109/JPROC.2023.3308088](https://doi.org/10.1109/JPROC.2023.3308088) |
| [2] | Neftci, E. O., Mostafa, H., & Zenke, F. (2019). "Surrogate Gradient Learning in Spiking Neural Networks." *IEEE Signal Processing Magazine*, 36(6), 51–63. → [doi:10.1109/MSP.2019.2931595](https://doi.org/10.1109/MSP.2019.2931595) |
| [3] | Deng, L., Wu, Y., Hu, X., Liang, L., et al. (2020). "Rethinking the Performance Comparison Between SNNs and ANNs." *Neural Networks*, 121, 294–307. → [doi:10.1016/j.neunet.2019.09.005](https://doi.org/10.1016/j.neunet.2019.09.005) |
| [4] | Fang, W., Chen, Y., Ding, J., Yu, Z., Masquelier, T., et al. (2023). "SpikingJelly: An Open-Source Machine Learning Infrastructure Platform for Spike-Based Intelligence." *Science Advances*, 9(40), eadi1480. → [doi:10.1126/sciadv.adi1480](https://doi.org/10.1126/sciadv.adi1480) |
| [5] | Orchard, G., Jayawant, A., Cohen, G. K., & Thakor, N. (2015). "Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades." *Frontiers in Neuroscience*, 9, 437. → [doi:10.3389/fnins.2015.00437](https://doi.org/10.3389/fnins.2015.00437) |
| [6] | Rueckauer, B., Lungu, I.-A., Hu, Y., Pfeiffer, M., & Liu, S.-C. (2017). "Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification." *Frontiers in Neuroscience*, 11, 682. → [doi:10.3389/fnins.2017.00682](https://doi.org/10.3389/fnins.2017.00682) |
| [7] | Wang, Z., Zhang, Y., Lian, S., Cui, X., Yan, R., & Tang, H. (2025). "Toward High-Accuracy and Low-Latency Spiking Neural Networks With Two-Stage Optimization." *IEEE TNNLS*, 36(2), 3189–3203. → [doi:10.1109/TNNLS.2023.3337176](https://doi.org/10.1109/TNNLS.2023.3337176) |
| [8] | He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*, pp. 770–778. → [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| [9] | Paszke, A., Gross, S., Massa, F., Lerer, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS 32*. → [arXiv:1912.01703](https://arxiv.org/abs/1912.01703) |
| [10] | Lemaire, E., Perotin, L., Courtois, T., & Masquelier, T. (2022). "An Analytical Estimation of Spiking Neural Networks Energy Efficiency." *IEEE ICASSP 2022*. → [doi:10.48550/arXiv.2210.13107](https://doi.org/10.48550/arXiv.2210.13107) |
| [11] | Dampfhoffer, M., Mesquida, T., Valentian, A., & Anghel, L. (2024). "Are SNNs Really More Energy-Efficient Than ANNs? An In-Depth Hardware-Aware Study." *IEEE TNNLS*, 35(9), 11906–11921. → [doi:10.1109/TNNLS.2023.3263008](https://doi.org/10.1109/TNNLS.2023.3263008) |


