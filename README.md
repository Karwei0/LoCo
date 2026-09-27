# LoCo: Local Competitive Learning for Interpretable Time Series

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

> LoCo is a lightweight and interpretable framework for multivariate time series anomaly detection, root cause analysis, and time-lagged causal discovery.

# 🧭 Overview

Multivariate time series anomaly detection is critical in streaming data scenarios such as cyber-physical systems, finance, and industrial monitoring. Beyond detecting when an anomaly occurs, operators need to know why it happened and where the root cause lies. Existing methods often rely on heavy deep architectures and only localize anomalous variables at the current timestamp, overlooking the temporal evolution of anomalies along causal chains.

LoCo is inspired by the physical concept of a light cone: just as we observe the sun as it was eight minutes ago, severe anomalies that trigger alerts are often shadows of inconspicuous upstream disturbances that occurred earlier. Motivated by two design principles—Finite Temporal Horizon and Causal Horizon, LoCo learns a sparse time-lagged causal graph directly from a single-layer predictor trained with local competitive learning.

# ✨ Key Features

- **Local Competitive Learning**: During backpropagation, each output variable updates only the top-\(k\) gradient-magnitude connections, forcing a sparse causal topology to emerge naturally.
- **Time-Lagged Causal Matrix**: The learned weight matrix serves as a fine-grained causal graph that encodes how past variables at specific lags influence current targets.
- **Interpretable by Design**: No post-hoc statistical tests. The causal graph is read directly from the converged weights.
- **Two Variants**: Linear LoCo provides transparent white-box weights; Kernel LoCo captures nonlinear dynamics via variable-wise Random Fourier Features while preserving causal attribution.
- **Lightweight and Efficient**: Minimal parameters, low memory overhead, fast inference, suitable for resource-constrained online deployment.
- **Unified Framework**: A single model simultaneously supports online anomaly detection, causal-driven root cause analysis, and time-lagged causal discovery.

# 🎯 Target Tasks

LoCo addresses three tasks in a unified workflow:

1. **Online Anomaly Detection (AD)**  
   Single-step forecasting via the learned predictor. Anomaly alarms are triggered based on prediction residuals.  
   *Metrics*: AUROC, AUPRC, Affiliation Precision/Recall/F1.

2. **Causal-Driven Root Cause Analysis (RCA)**  
   Upon detection, LoCo identifies the most deviated variable at the current timestamp as the conventional root cause, and further traces backward along the learned time-lagged causal matrix to earlier upstream root causes.  
   *Metrics*: AC@K, AC*@K, Avg@K, Avg*@K.

3. **Time-Lagged Causal Discovery (CD)**  
   The same time-lagged matrix can be compressed into a global \(N \times N\) causal graph for conventional causal discovery tasks.  
   *Metrics*: AUROC, Individual AUROC (InROC).

# 📁 Repository Structure

```txt
LoCo/
├── cd/                     # Scripts and code for causal discovery task
├── data_provider/          # Data loading and preprocessing (dataset classes, dataloaders)
├── datasets/               # Raw or processed benchmark datasets
├── experiments/            # Configuration files and experiment launchers
├── imgs/                   # Figures and visualizations
├── layers/                 # Core modules.
├── models/                 # Main LoCo model definition (Linear & Kernel)
├── rca/                    # Scripts and code for root cause analysis task
├── scripts/                # Utility bash scripts for training, evaluation,
├── utils/                  # Helper functions (metrics, logger, visualization)
├── config.py               # Global configuration
├── run.py                  # Main entry point for training and testing
└── README.md               # This file
```

# 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/karweii/LoCo.git
cd LoCo
```

2. Create a conda environment:
```bash
conda create -n loco python=3.8
conda activate loco
```

3. Install PyTorch (adjust CUDA version as needed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

# 📊 Quick Start

## Anomaly Detection (AD)

1. **Data preparation**  
   Download the datasets MSL, SMAP, PSM, SWaT, SMD, GECCO, Creditcard, SWAN from  
   [Most AD datasets](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).  
   For CICIDS, download from  
   [CICIDS dataset](https://drive.google.com/file/d/1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT/view?usp=drive_link).  
   Then run:
   ```bash
   python datasets/process.py
   ```

2. **Train a model**  
   For example, to train Linear LoCo on SWaT:
   ```bash
   bash scripts/linear_loco/swat.sh
   ```
   Kernel LoCo scripts are under `scripts/kernel_loco/`.

## Root Cause Analysis (RCA)

We build upon the [AERCA](https://github.com/hanxiao0607/AERCA) repository, which provides a complete RCA pipeline.  
- Place the LoCo core implementation under `models/` of the AERCA repository.  
- Related scripts are provided in `rca/` of this repository.  
- Follow the instructions in the AERCA repository to run RCA tasks.

## Causal Discovery (CD)

We adopt two repositories from CausalRivers:
- [causalrivers](https://github.com/CausalRivers/causalrivers) for dataset processing.
- [experiments](https://github.com/CausalRivers/experiments) for benchmarking.

- Place the LoCo core implementation under `causal_discovery_zoo/methods` of the `experiments` repository.  
- Related scripts are provided in `cd/` of this repository.  
- Follow the instructions in the CausalRivers repositories to run CD tasks.

# 🙏 Acknowledgements

We thank the following projects for providing datasets, code, and inspiration:

- [DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector.git) for cleaned versions of the datasets.
- [TAB](https://github.com/decisionintelligence/TAB.git) for partial AD datasets and implementations of AD-specific and generative baselines.
- [AERCA](https://github.com/hanxiao0607/AERCA) for RCA datasets and codebase.
- [causalrivers-dataset](https://github.com/CausalRivers/causalrivers) for the ready-to-use causal discovery dataset.
- [causalrivers-benchmark](https://github.com/CausalRivers/experiments) for the complete benchmark for comparison.

# 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{loco2026,
  title={LoCo: Local Competitive Learning for Interpretable Multivariate Time Series Anomaly Detection},
  author={Anonymous},
  journal={Under review},
  year={2026}
}
```
