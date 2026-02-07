# Wavelet Packet-Based Diffusion Model for Ground Motion Generation

<div align="center">

**Multi-Conditional Diffusion Model for Ground Motion Generation with Energy and Spectral Matching**

[Paper](https://www.researchgate.net/publication/400550050_Wavelet_Packet-Based_Diffusion_Model_for_Ground_Motion_Generation_with_Multi-Conditional_Energy_and_Spectral_Matching) | [Pre-trained Weights](https://drive.google.com/drive/folders/1zKBlBUlEgbvNbO-ddcceOnB1O8VtSan2?usp=drive_link)

</div>

---

## Introduction

This repository implements a diffusion model-based framework for earthquake ground motion generation that simultaneously matches response spectra and temporal energy evolution characteristics. The method employs **Daubechies-6 (Db6) wavelet packet decomposition** for signal representation, combined with the **Elucidating Diffusion Model (EDM)** and **second-order Heun sampler** to achieve efficient, high-quality ground motion synthesis.

### Key Features

- **Multi-Conditional Constraints**: Simultaneous control of response spectrum, Arias intensity, significant duration, temporal energy parameters, and Husid curve
- **Wavelet Packet Representation**: Precise waveform reconstruction via orthogonal filter banks (reconstruction error ~10⁻¹⁴), avoiding iterative phase recovery
- **Transformer Conditional Encoder**: Fuses heterogeneous conditional information and injects it into the diffusion process via cross-attention mechanism
- **Efficient Sampling**: Second-order Heun sampler requires only 25 steps (50 network evaluations) to generate high-quality samples
- **Uncertainty Quantification**: Supports conditional diversity sampling for structural response uncertainty analysis

### Methodology Overview

The framework comprises three core components:

1. **Wavelet Packet Decomposition and Reconstruction**: Decomposes 16,384-point ground motion time histories into a 128×128 time-frequency representation, with independent normalization for each subband
2. **Conditional Encoder**: Transformer-based encoder fuses vector conditions (response spectrum, Husid curve) and scalar intensity parameters (Arias intensity, T₅, D₅₋₉₅, Eₜₕ)
3. **EDM Denoising Network**: U-Net architecture with cross-attention mechanism, trained in normalized wavelet packet coefficient space

Compared to STFT+Griffin-Lim-based methods, wavelet packet representation offers the following advantages:
- Reconstruction precision improved to machine precision level
- Single-pass forward reconstruction without iterative phase recovery
- Linear transformation with perfect phase preservation

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.5+
- CUDA 11.8+ (for GPU training)

### Dependencies

It is recommended to use the provided conda environment configuration file:

```bash
conda env create -f torch2.5-linux.yml
conda activate wp-diffusion
```

Or install dependencies manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy h5py pywavelets tqdm
```

---

## Data Preparation

### NGA-West2 Database

This project uses the NGA-West2 ground motion database for training and testing.

**Data Access**:

Please obtain data from the Pacific Earthquake Engineering Research Center (PEER):

```
https://ngawest2.berkeley.edu/
```

**Data Preprocessing**:

After downloading the raw waveforms and metadata, use the scripts in the `build_datasets/` directory to construct HDF5 datasets:

```bash
python build_datasets/build_hdf5.py --input <NGA_DIR> --output <OUTPUT_DIR>
```

For detailed information on the original dataset structure, please refer to:
- `build_datasets/NGA_DATA.md`

**Dataset Split**:
- Training set: 17,784 records (90%)
- Test set: 1,976 records (10%)

**Preprocessing Steps**:
1. Resample to 100 Hz (low-sampling-rate records upsampled via linear interpolation, high-sampling-rate records low-pass filtered and downsampled)
2. Standardize duration to 163.84 seconds (16,384 points) via truncation or zero-padding
3. Compute response spectra and key intensity parameters, followed by normalization

---

## Usage

### Configuration

Runtime configuration is controlled by module-level constants in `Main.py` and configuration classes in `config.py`:

**Main.py Configuration Options**:
- `STATE`: `"train"` or `"test"` - Training/testing mode
- `COND_CONFIG_ID`:
  - `0` - Response spectrum only
  - `1` - Response spectrum + intensity parameters
- `RANDOM_INFERENCE_MODE`: `False` (default) - Deterministic sampling

**config.py Configuration Options**:
- `DatasetConfig.DATASET_NAME` - Dataset name
- `PathConfig.DATASETS_DIR` - Dataset path
- `PathConfig.WEIGHTS_DIR` - Weights save/load path

### Training

**Training Parameters**:
- Optimizer: Adam (lr=1×10⁻⁴)
- Batch size: 110
- EMA decay rate: 0.999
- Training epochs: 100 epochs
- Mixed precision: BF16

```bash
# Set in Main.py
STATE = "train"

# Execute
python Main.py
```

### Testing/Inference

#### Using Pre-trained Weights

Download pre-trained weights from Google Drive:

```
https://drive.google.com/drive/folders/1zKBlBUlEgbvNbO-ddcceOnB1O8VtSan2?usp=drive_link
```

Place the weight files in the directory specified by `PathConfig.WEIGHTS_DIR`.

#### Running Inference

```bash
# Set in Main.py
STATE = "test"

# Execute
python Main.py
```

---

## Project Structure

```
wavelet-packet-diffusion-gm/
├── Main.py                              # Main entry point for training/testing
├── config.py                            # Configuration classes (paths, dataset, model)
├── train_eval.py                        # Training and evaluation logic
├── torch2.5-linux.yml                   # Conda environment configuration
├── build_datasets/                      # Dataset construction scripts
│   ├── build_dataset1_nga_100hz.py     # Step 1: Build NGA-West2 HDF5 dataset
│   ├── build_dataset2_pga_cdf.py       # Step 2: Build PGA CDF for normalization
│   └── NGA_DATA.md                     # NGA-West2 data documentation
├── diffusion/                           # Core diffusion model implementation
│   ├── edm.py                          # Elucidating Diffusion Model (EDM)
│   ├── unet.py                         # U-Net architecture with cross-attention
│   ├── blocks.py                       # Neural network building blocks
│   ├── nn.py                           # Neural network utilities
│   ├── representation.py               # Wavelet packet decomposition/reconstruction
│   ├── dataset.py                      # PyTorch dataset classes
│   ├── data_utils.py                   # Data loading utilities
│   ├── seismic_metrics.py              # Intensity parameter calculations
│   └── utils.py                        # General utilities
├── datasets/                            # Dataset storage directory (user-created)
├── outputs/                             # Training outputs and generated results
│   └── results/                        # Inference results by configuration
└── README.md                           # This document
```

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ding2025wavelet,
  title={Wavelet Packet-Based Diffusion Model for Ground Motion Generation with Multi-Conditional Energy and Spectral Matching},
  author={Ding, Yi and Chen, Su and Hu, Jinjun and Hu, Xiaohu and Zhao, Qingxu and Li, Xiaojun},
  journal={Preprint},
  year={2025}
}
```

