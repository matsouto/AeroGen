# Generative Airfoil Models

A deep learning project for generating novel airfoil designs using generative models including GANs (Generative Adversarial Networks) and VAEs (Variational Autoencoders). This repository implements multiple architectures for airfoil generation and optimization.

## Overview

This project explores generative modeling techniques to create new aerodynamic airfoil profiles. The implementations include:

- **CST GAN**: Generative model using Class-Shape Transformation (CST) parameterization
- **CST VAE**: Variational Autoencoder with CST coefficients
- **Plain GAN**: Traditional GAN architecture for 79-point and 149-point airfoil representations
- **Export & Inference**: ONNX model export for production deployment

## Project Structure

```
├── src/                           # Source code modules
│   ├── airfoil/                   # Airfoil utilities and computations
│   │   ├── airfoil_modifications.py
│   │   ├── compute_airfoil_quality.py
│   │   └── helpers.py
│   ├── gan/                       # GAN architectures
│   │   ├── generator.py
│   │   └── discriminator.py
│   ├── vae/                       # VAE architectures
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── vae.py
│   │   └── train.py
│   ├── layers/                    # Custom neural network layers
│   ├── utils.py                   # Utility functions
│   ├── plotting.py                # Visualization utilities
│   └── export_onnx.py             # ONNX export functionality
│
├── notebook/                      # Jupyter notebooks for workflows
│   ├── preprocessing/
│   │   ├── preprocessing.ipynb    # Data preprocessing pipeline
│   │   └── modeling.ipynb
│   ├── training/                  # Training scripts
│   │   ├── cstgan/
│   │   ├── cstvae/
│   │   └── gan/
│   └── validation/                # Model validation
│       ├── vae_validation.ipynb
│       └── vae_validation_2.ipynb
│
├── data/                          # datasets
│   ├── airfoils/                  # Raw airfoil data
│   └── processed/                 # Processed datasets
│       ├── airfoil_dataset.json
│       ├── airfoil_dataset_40.json
│       ├── train_kulfan_dataset_75.json
│       └── val_kulfan_dataset_75.json
│
├── models/                        # Trained model checkpoints
│   ├── cstgan/
│   ├── cstvae/
│   ├── plaingan_79_points/
│   └── plaingan_149_points/
│
├── exported_models/               # ONNX exported models
│   └── 20260221-165429_decoder.onnx
│
├── test/                          # Testing scripts
│   └── small_overfit_test.py
│
├── requirements.txt               # Python dependencies
└── sketch.ipynb                   # Exploratory notebook
```

## Key Features

### Generative Models

1. **CST GAN** - Uses Class-Shape Transformation coefficients (dimensionality: 24 coefficients)
2. **CST VAE** - Variational autoencoder with CST parameterization, includes latent space regularization
3. **Plain GAN** - Direct coordinate-based generation with support for both 79-point and 149-point formats

### Airfoil Utilities

- **Quality Computation**: Compute aerodynamic quality metrics (VOD - Variation of Differences, smoothness analysis)
- **Airfoil Modifications**: Transform and modify airfoil geometries
- **Helpers**: Common operations for airfoil manipulation

### Training & Validation

- Support for Weights & Biases (wandb) experiment tracking
- Multiple preprocessing pipelines for different representations
- Validation notebooks for model evaluation

### Export & Deployment

- ONNX format export for cross-platform deployment
- Inference-ready models for production environments

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Generative_airfoil_models
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

Key packages:
- **TensorFlow & Keras** (3.13.2) - Deep learning framework
- **NumPy** (2.1.3) - Numerical computing
- **Pandas** (3.0.0) - Data manipulation
- **Scikit-learn** (1.8.0) - Machine learning utilities
- **SciPy** (1.17.0) - Scientific computing
- **Matplotlib & Seaborn** - Visualization
- **Plotly** (6.5.2) - Interactive plotting
- **AeroSandbox** (4.2.9) - Aerodynamic analysis
- **NeuralFoil** (0.3.2) - Neural airfoil analysis
- **ONNX** (1.17.0) - Model interchange format

For complete list, see [requirements.txt](requirements.txt).

## Usage

### Data Preprocessing

Start with the preprocessing notebooks to prepare your airfoil data:

```bash
jupyter notebook notebook/preprocessing/preprocessing.ipynb
```

### Training VAE Models

Train a CST VAE model:

```bash
# See notebook/training/cstvae/ for training scripts
python -m src.vae.train
```

### Training GAN Models

Train Plain GAN or CST GAN models:

```bash
# See notebook/training/gan/ or notebook/training/cstgan/ for training scripts
```

### Model Validation

Evaluate trained models:

```bash
jupyter notebook notebook/validation/vae_validation.ipynb
```

### Export to ONNX

Export trained models for production deployment:

```bash
python src/export_onnx.py
```

## Model Architectures

### VAE Architecture

**Encoder:**
- Flatten layer
- Dense(256) + BatchNorm + LeakyReLU
- Dense(512) + BatchNorm + LeakyReLU
- Latent space: z_mean and z_log_var (default latent_dim=16)

**Decoder:**
- Mirrors encoder architecture
- Reconstructs airfoil parameters

### GAN Architecture

**Generator:**
- Takes noise vector as input
- Generates valid airfoil representations

**Discriminator:**
- Classifies real vs. generated airfoils
- Binary classification output

## Configuration

### Experiment Tracking

The project integrates with Weights & Biases for experiment tracking. Logs are stored in the `wandb/` directory.

To enable wandb tracking:
1. Create account at [weights-and-biases.ai](https://www.wandb.ai/)
2. Add your API key to `API_KEY.txt`
3. Models will automatically log metrics during training

## Data Formats

### Airfoil Representations

1. **CST Coefficients**: 24-dimensional vectors (12 coefficients for upper/lower surfaces)
2. **Coordinate Points (79-point)**: Direct (x, y) coordinates at 79 locations
3. **Coordinate Points (149-point)**: Direct (x, y) coordinates at 149 locations

### Input Data

- `airfoil_dataset.json`: Main dataset with airfoil geometries
- `airfoil_dataset_40.json`: Subset with 40 points per side
- `train_kulfan_dataset_75.json`: Training set with 75 points per side
- `val_kulfan_dataset_75.json`: Validation set with 75 points per side

## Model Checkpoints

Trained models are stored in `models/` with timestamps:

- `models/cstvae/20260202-002116/` - Timestamped VAE checkpoints
- `models/plaingan_79_points/` - 79-point GAN models
- `models/plaingan_149_points/` - 149-point GAN models

## Exported Models

Production-ready models in [exported_models/](exported_models/) directory in ONNX format for deployment.

## Testing

Run tests to verify the setup:

```bash
python test/small_overfit_test.py
```

## Notebooks Guide

| Notebook | Purpose |
|----------|---------|
| `sketch.ipynb` | Exploratory analysis |
| `preprocessing/preprocessing.ipynb` | Data preprocessing pipeline |
| `preprocessing/modeling.ipynb` | Modeling setup |
| `validation/vae_validation.ipynb` | VAE model evaluation |

## Output & Results

Generated airfoils are validated and can be:
- Visualized using matplotlib/plotly
- Exported as ONNX models
- Analyzed for aerodynamic properties
- Used as training data for aerodynamic optimization

## References

- CST (Class-Shape Transformation) Parameterization
- Variational Autoencoders in Generative Modeling
- Generative Adversarial Networks
- ONNX Model Interchange Format

---

**Last Updated:** February 2026
**Python Version:** 3.8+
**Framework:** TensorFlow/Keras 3.13+
