# Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for training

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (>=2.0.0) and torchvision (>=0.15.0)
- NumPy (>=1.24.0)
- Matplotlib (>=3.7.0)
- Pillow (>=9.5.0)
- tqdm (>=4.65.0)
- scikit-learn (>=1.2.0)

### 3. Verify Installation

Run the test script to verify everything is working:

```bash
python test_models.py
```

Expected output:
```
Testing Model Architectures
==================================================

1. Testing CNN Classifier...
   ✓ Model initialized successfully
   ✓ Forward pass successful
   ✓ Prediction method works
   ✓ CNN Classifier: ALL TESTS PASSED

2. Testing Variational Autoencoder...
   ✓ Model initialized successfully
   ✓ Forward pass successful
   ...
   ✓ VAE: ALL TESTS PASSED

...
✓ ALL TESTS PASSED!
```

## GPU Support

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Install PyTorch with CUDA Support

If CUDA is not available but you have an NVIDIA GPU, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for more installation options.

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Make sure you've activated your virtual environment and installed requirements.txt

### Issue: Training is very slow
**Solution**: 
- Check if GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size if running out of memory
- Use CPU for testing by setting `device='cpu'` in training scripts

### Issue: "RuntimeError: CUDA out of memory"
**Solution**:
- Reduce batch size in training scripts
- Reduce image size (e.g., from 64 to 32)
- Close other GPU-intensive applications

### Issue: Dataset download fails
**Solution**:
- Check your internet connection
- The CelebA dataset is large (~1.4 GB), ensure sufficient disk space
- Download may be slow, be patient
- If download fails, try deleting the `data/` folder and running again

## Directory Structure After Setup

After installation and first training run, your directory will look like:

```
machine_learning2_exam/
├── data/                    # CelebA dataset (created on first run)
│   └── celeba/
├── checkpoints/            # Saved models (created during training)
│   ├── cnn_best.pth
│   └── vae_best.pth
├── results/                # Generated images (created during training)
│   ├── reconstruction_epoch_*.png
│   ├── latent_traversal.png
│   └── interpolation.png
├── venv/                   # Virtual environment (if created)
└── [source files]
```

## Next Steps

Once setup is complete:

1. **Test the models**: `python test_models.py`
2. **Train the CNN**: `python train_cnn.py`
3. **Train the VAE**: `python train_vae.py`
4. **Run examples**: `python example_usage.py`

See the main [README.md](README.md) for detailed usage instructions.
