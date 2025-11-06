# Exam Guide: CNN and VAE on CelebA with Latent Traversal

This guide provides a comprehensive overview of the project for exam purposes.

## Project Overview

This project demonstrates mastery of:
1. **Deep Learning Architectures**: CNN for classification, VAE for generation
2. **Multi-label Classification**: Predicting 40 attributes simultaneously
3. **Generative Modeling**: Learning to generate realistic images
4. **Latent Space Analysis**: Understanding and exploring learned representations

## Key Concepts

### 1. Convolutional Neural Network (CNN)

**Architecture Components**:
- **Convolutional Layers**: Extract spatial features from images
- **Batch Normalization**: Stabilizes training by normalizing layer inputs
- **Pooling Layers**: Reduce spatial dimensions and computational cost
- **Dropout**: Prevents overfitting by randomly dropping neurons during training
- **Fully Connected Layers**: Combine features for final predictions

**Multi-label Classification**:
- Each image has multiple binary attributes (e.g., Male, Smiling, Eyeglasses)
- Use Binary Cross-Entropy loss (BCEWithLogitsLoss)
- Sigmoid activation to get probabilities for each attribute independently

### 2. Variational Autoencoder (VAE)

**Key Components**:

1. **Encoder**: Compresses input images to latent representations
   - Outputs mean (μ) and log-variance (log σ²) for each latent dimension
   
2. **Latent Space**: Continuous vector representation
   - Follows a standard normal distribution N(0, 1)
   - Enables smooth interpolation and controlled generation
   
3. **Reparameterization Trick**: Enables backpropagation through sampling
   - z = μ + σ * ε, where ε ~ N(0, 1)
   - Gradient flows through μ and σ, not through random ε
   
4. **Decoder**: Reconstructs images from latent vectors
   - Uses transposed convolutions to upsample

**Loss Function**:
```
Loss = Reconstruction Loss + β × KL Divergence

- Reconstruction Loss: MSE between original and reconstructed images
- KL Divergence: Regularizes latent space to match N(0, 1)
- β: Weight parameter (β-VAE allows controlling disentanglement)
```

**Beta-VAE**:
- β > 1 encourages more disentangled representations
- β < 1 focuses more on reconstruction quality
- Trade-off between reconstruction and disentanglement

### 3. Latent Space Traversal

**Why Explore Latent Space?**
- Understand what features the model learned
- Verify semantic meaning of latent dimensions
- Generate controlled variations of images

**Techniques**:

1. **Linear Interpolation**: Smooth transition between two images
   - z(t) = (1-t) × z₁ + t × z₂, where t ∈ [0, 1]
   
2. **Single Dimension Traversal**: Vary one dimension, fix others
   - Reveals what each dimension controls (e.g., age, gender, smile)
   
3. **Random Sampling**: Generate new faces from noise
   - z ~ N(0, 1), then decode to image

## Implementation Details

### CNN Architecture

```python
Input: 3 × 64 × 64
↓ Conv(32) + BN + Pool
↓ Conv(64) + BN + Pool
↓ Conv(128) + BN + Pool
↓ Conv(256) + BN + Pool
↓ Conv(512) + BN + Pool
↓ Flatten
↓ FC(1024) + Dropout
↓ FC(512) + Dropout
↓ FC(40)
Output: 40 logits (one per attribute)
```

### VAE Architecture

**Encoder**:
```python
Input: 3 × 64 × 64
↓ Conv(32, stride=2)
↓ Conv(64, stride=2)
↓ Conv(128, stride=2)
↓ Conv(256, stride=2)
↓ Flatten → μ, log(σ²)
Output: 128-dim latent vector
```

**Decoder**:
```python
Input: 128-dim latent vector
↓ FC → Reshape to 256 × 4 × 4
↓ ConvTranspose(128, stride=2)
↓ ConvTranspose(64, stride=2)
↓ ConvTranspose(32, stride=2)
↓ ConvTranspose(3, stride=2)
Output: 3 × 64 × 64
```

## Training Strategy

### CNN Training
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 20
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Validation**: Monitor accuracy and loss

### VAE Training
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 50
- **Beta**: 1.0 (standard VAE)
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Validation**: Monitor reconstruction quality

## Expected Results

### CNN Performance
- **Overall Accuracy**: ~85-90%
- **Easy Attributes**: Male (>95%), Smiling (>90%)
- **Hard Attributes**: Bald (<70%), Gray_Hair (<75%)
- **Reason**: Class imbalance in dataset

### VAE Performance
- **Early Epochs (1-10)**: Blurry reconstructions
- **Mid Training (10-30)**: Recognizable faces, some details
- **Late Training (30-50)**: High-quality reconstructions
- **Latent Space**: Smooth interpolations, meaningful traversals

## Key Insights

1. **CNN learns discriminative features** for classification
2. **VAE learns generative model** of face distribution
3. **Latent space structure** reveals learned semantic concepts
4. **Disentanglement** varies by training (beta value, architecture)
5. **Trade-offs** exist between reconstruction quality and regularization

## Common Questions

**Q: Why use VAE instead of regular autoencoder?**
A: VAE enforces continuous, structured latent space (via KL divergence), enabling smooth interpolation and controlled generation.

**Q: What does each latent dimension represent?**
A: Dimensions often learn interpretable features (age, gender, pose, lighting) but this depends on training and isn't guaranteed.

**Q: Why normalize images to [-1, 1]?**
A: Matches tanh output activation and helps training stability. Common practice in image generation.

**Q: How to improve results?**
A: 
- Train longer (more epochs)
- Tune beta in VAE for better disentanglement
- Use larger latent dimension for more capacity
- Add more layers for more complex patterns
- Try different architectures (e.g., ResNet blocks)

## Evaluation Criteria

For exam purposes, understanding of:
1. ✅ CNN architecture and multi-label classification
2. ✅ VAE theory and reparameterization trick
3. ✅ Loss functions (BCE, reconstruction + KL divergence)
4. ✅ Latent space exploration techniques
5. ✅ Implementation in PyTorch
6. ✅ Training procedures and hyperparameters
7. ✅ Result interpretation and analysis

## References for Further Study

- **VAE Paper**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **Beta-VAE**: [Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl)
- **CelebA Dataset**: [Large-scale CelebFaces Attributes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

**Good luck with your exam!** 🎓
