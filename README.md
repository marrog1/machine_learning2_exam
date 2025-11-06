# Machine Learning 2 Exam: CNN and VAE on CelebA

This repository contains implementations of a **Convolutional Neural Network (CNN)** for attribute classification and a **Variational Autoencoder (VAE)** for image generation, both trained on the CelebA (CelebFaces Attributes) dataset. The project also includes **latent space traversal** functionality to explore the learned representations of the VAE.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training the CNN](#training-the-cnn)
  - [Training the VAE](#training-the-vae)
  - [Running Examples](#running-examples)
- [Models](#models)
  - [CNN Classifier](#cnn-classifier)
  - [Variational Autoencoder](#variational-autoencoder)
- [Latent Space Exploration](#latent-space-exploration)
- [Results](#results)

## 🎯 Overview

This project demonstrates:

1. **Multi-label Classification**: A CNN that predicts 40 facial attributes (e.g., Smiling, Male, Eyeglasses) from CelebA images
2. **Generative Modeling**: A VAE that learns to generate realistic face images
3. **Latent Space Analysis**: Tools to explore and visualize the VAE's learned latent representations through traversal and interpolation

## ✨ Features

- **CNN Classifier**
  - Deep convolutional architecture with batch normalization
  - Multi-label binary classification for 40 attributes
  - Training with validation monitoring and model checkpointing
  
- **Variational Autoencoder**
  - Convolutional encoder-decoder architecture
  - Reparameterization trick for backpropagation through sampling
  - Beta-VAE support for controlled disentanglement
  - Random sampling from learned distribution
  
- **Latent Space Exploration**
  - Linear interpolation between latent vectors
  - Single-dimension traversal to understand individual latent factors
  - Multi-dimension traversal visualization
  - Image-to-image interpolation through latent space

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/marrog1/machine_learning2_exam.git
cd machine_learning2_exam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which contains:
- 202,599 face images
- 40 binary attribute annotations per image
- 10,177 unique identities

The dataset will be automatically downloaded when you run the training scripts for the first time.

## 📁 Project Structure

```
machine_learning2_exam/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── data_utils.py            # Data loading and preprocessing utilities
├── cnn_classifier.py        # CNN model architecture
├── vae.py                   # VAE model architecture and loss function
├── latent_traversal.py      # Latent space exploration utilities
├── train_cnn.py             # CNN training script
├── train_vae.py             # VAE training script
├── example_usage.py         # Example scripts demonstrating usage
├── checkpoints/             # Saved model checkpoints (created during training)
├── results/                 # Generated visualizations (created during training)
└── data/                    # CelebA dataset (downloaded automatically)
```

## 🚀 Usage

### Training the CNN

Train the CNN classifier for attribute prediction:

```bash
python train_cnn.py
```

**Key parameters** (can be modified in the script):
- `num_epochs`: Number of training epochs (default: 20)
- `batch_size`: Batch size (default: 64)
- `learning_rate`: Initial learning rate (default: 0.001)
- `image_size`: Size to resize images (default: 64)
- `device`: Training device ('cuda' or 'cpu')

The trained model will be saved to `checkpoints/cnn_best.pth`.

### Training the VAE

Train the Variational Autoencoder:

```bash
python train_vae.py
```

**Key parameters** (can be modified in the script):
- `num_epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 64)
- `learning_rate`: Initial learning rate (default: 0.001)
- `latent_dim`: Dimensionality of latent space (default: 128)
- `image_size`: Size to resize images (default: 64)
- `beta`: Weight for KL divergence, beta-VAE (default: 1.0)
- `device`: Training device ('cuda' or 'cpu')

The trained model will be saved to `checkpoints/vae_best.pth`, and reconstruction samples will be saved to the `results/` directory every 5 epochs.

### Running Examples

After training both models, run the example scripts to see them in action:

```bash
python example_usage.py
```

This will:
1. Load the trained CNN and predict attributes for test images
2. Perform latent space traversal with the VAE
3. Generate interpolations between images
4. Generate random samples from the VAE

All visualizations will be saved to the `results/` directory.

## 🧠 Models

### CNN Classifier

The CNN architecture consists of:
- **5 Convolutional Blocks**: Each with Conv2D, BatchNorm, ReLU, and MaxPooling
- **Feature Maps**: 32 → 64 → 128 → 256 → 512 channels
- **Fully Connected Layers**: 3 layers with dropout for regularization
- **Output**: 40 sigmoid activations for multi-label classification

**Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

### Variational Autoencoder

The VAE consists of:

**Encoder**:
- 4 convolutional layers with stride 2 (downsampling)
- Outputs mean (μ) and log-variance (log σ²) vectors

**Latent Space**:
- Reparameterization: z = μ + σ * ε, where ε ~ N(0, 1)
- Default dimensionality: 128

**Decoder**:
- 4 transposed convolutional layers (upsampling)
- Reconstructs images from latent vectors

**Loss Function**: 
- Reconstruction Loss (MSE) + β × KL Divergence
- KL Divergence regularizes the latent space to match a standard normal distribution

## 🔍 Latent Space Exploration

### Linear Interpolation

Smoothly interpolate between two latent vectors:
```python
from latent_traversal import linear_interpolation

images = linear_interpolation(vae, z1, z2, num_steps=10)
```

### Single Dimension Traversal

Vary a single latent dimension while keeping others fixed:
```python
from latent_traversal import latent_traversal

images = latent_traversal(vae, base_z, dim_idx=5, min_val=-3, max_val=3)
```

### Multi-Dimension Traversal

Explore multiple latent dimensions simultaneously:
```python
from latent_traversal import random_latent_traversal, visualize_multiple_traversals

traversals, base_z = random_latent_traversal(vae, latent_dim=128, num_dims=6)
visualize_multiple_traversals(traversals, save_path='traversal.png')
```

### Image-to-Image Interpolation

Encode two real images and interpolate through latent space:
```python
from latent_traversal import encode_and_interpolate

interpolated = encode_and_interpolate(vae, img1, img2, num_steps=10)
```

## 📈 Results

After training, you can expect:

**CNN Classifier**:
- Validation accuracy: ~85-90% (varies by attribute)
- Better performance on attributes with balanced classes
- Some attributes (e.g., "Male", "Smiling") are easier to predict than others

**VAE**:
- Realistic face reconstructions after 30-50 epochs
- Smooth latent space interpolations
- Disentangled representations (especially with beta > 1)
- Generated samples show diversity while maintaining face structure

Example visualizations are saved in the `results/` directory:
- `reconstruction_epoch_*.png`: Original vs. reconstructed images
- `latent_traversal.png`: Effects of varying different latent dimensions
- `interpolation.png`: Smooth transitions between images
- `random_samples.png`: Randomly generated faces

## 📚 References

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) (Higgins et al., 2017)

## 📝 License

This project is for educational purposes as part of a Machine Learning exam.

## 👥 Author

Created for Machine Learning 2 Exam