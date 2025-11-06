"""
Basic test script to verify model architectures without training.
Tests model initialization and forward passes with dummy data.
"""
import torch
import sys

print("Testing Model Architectures")
print("=" * 50)

# Test CNN Classifier
print("\n1. Testing CNN Classifier...")
try:
    from cnn_classifier import CelebACNN
    
    # Initialize model
    model = CelebACNN(num_attributes=40, image_size=64)
    print("   ✓ Model initialized successfully")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (4, 40), f"Expected (4, 40), got {output.shape}"
    print(f"   ✓ Forward pass successful: input {dummy_input.shape} -> output {output.shape}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    assert predictions.shape == (4, 40), f"Expected (4, 40), got {predictions.shape}"
    print(f"   ✓ Prediction method works: {predictions.shape}")
    
    print("   ✓ CNN Classifier: ALL TESTS PASSED")
    
except Exception as e:
    print(f"   ✗ CNN Classifier failed: {e}")
    sys.exit(1)

# Test VAE
print("\n2. Testing Variational Autoencoder...")
try:
    from vae import VAE, vae_loss
    
    # Initialize model
    model = VAE(latent_dim=128, image_size=64)
    print("   ✓ Model initialized successfully")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    x_recon, mu, logvar = model(dummy_input)
    
    # Check output shapes
    assert x_recon.shape == (4, 3, 64, 64), f"Expected (4, 3, 64, 64), got {x_recon.shape}"
    assert mu.shape == (4, 128), f"Expected (4, 128), got {mu.shape}"
    assert logvar.shape == (4, 128), f"Expected (4, 128), got {logvar.shape}"
    print(f"   ✓ Forward pass successful")
    print(f"     - Reconstruction: {x_recon.shape}")
    print(f"     - Mu: {mu.shape}")
    print(f"     - Logvar: {logvar.shape}")
    
    # Test encode
    mu_enc, logvar_enc = model.encode(dummy_input)
    assert mu_enc.shape == (4, 128)
    print(f"   ✓ Encode method works: {mu_enc.shape}")
    
    # Test reparameterize
    z = model.reparameterize(mu, logvar)
    assert z.shape == (4, 128)
    print(f"   ✓ Reparameterize method works: {z.shape}")
    
    # Test decode
    decoded = model.decode(z)
    assert decoded.shape == (4, 3, 64, 64)
    print(f"   ✓ Decode method works: {decoded.shape}")
    
    # Test sampling
    samples = model.sample(8, device='cpu')
    assert samples.shape == (8, 3, 64, 64)
    print(f"   ✓ Sample method works: {samples.shape}")
    
    # Test loss function
    total_loss, recon_loss, kl_div = vae_loss(x_recon, dummy_input, mu, logvar)
    assert total_loss.item() >= 0
    print(f"   ✓ Loss function works")
    print(f"     - Total loss: {total_loss.item():.4f}")
    print(f"     - Recon loss: {recon_loss.item():.4f}")
    print(f"     - KL div: {kl_div.item():.4f}")
    
    print("   ✓ VAE: ALL TESTS PASSED")
    
except Exception as e:
    print(f"   ✗ VAE failed: {e}")
    sys.exit(1)

# Test data utilities
print("\n3. Testing Data Utilities...")
try:
    from data_utils import get_attribute_names
    
    attr_names = get_attribute_names()
    assert len(attr_names) == 40, f"Expected 40 attributes, got {len(attr_names)}"
    print(f"   ✓ Attribute names loaded: {len(attr_names)} attributes")
    print(f"   ✓ Sample attributes: {attr_names[:5]}")
    
    print("   ✓ Data Utilities: ALL TESTS PASSED")
    
except Exception as e:
    print(f"   ✗ Data Utilities failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✓ ALL TESTS PASSED!")
print("\nAll model architectures are working correctly.")
print("The models can be trained using:")
print("  - python train_cnn.py")
print("  - python train_vae.py")
print("\nNote: Training requires the dependencies in requirements.txt")
