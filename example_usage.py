"""
Example usage of CNN and VAE with latent traversal on CelebA dataset.
"""
import torch
import os

from cnn_classifier import CelebACNN
from vae import VAE
from data_utils import get_celeba_dataloader, get_attribute_names
from latent_traversal import (
    random_latent_traversal,
    visualize_multiple_traversals,
    encode_and_interpolate,
    visualize_interpolation
)


def load_trained_cnn(checkpoint_path, device='cuda'):
    """
    Load a trained CNN model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded CNN model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = CelebACNN(num_attributes=40, image_size=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded CNN model from {checkpoint_path}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return model


def load_trained_vae(checkpoint_path, device='cuda'):
    """
    Load a trained VAE model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded VAE model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    latent_dim = checkpoint.get('latent_dim', 128)
    image_size = checkpoint.get('image_size', 64)
    model = VAE(latent_dim=latent_dim, image_size=image_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded VAE model from {checkpoint_path}")
    print(f"Latent dimension: {latent_dim}")
    
    return model


def predict_attributes(cnn_model, images, device='cuda'):
    """
    Predict attributes for given images using CNN.
    
    Args:
        cnn_model: Trained CNN model
        images: Input images tensor
        device: Device to run on
        
    Returns:
        Predicted attributes
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    cnn_model.eval()
    with torch.no_grad():
        images = images.to(device)
        predictions = cnn_model.predict(images)
    
    return predictions


def example_cnn_prediction():
    """
    Example of using CNN for attribute prediction.
    """
    print("\n=== CNN Attribute Prediction Example ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint_path = 'checkpoints/cnn_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the CNN first using train_cnn.py")
        return
    
    model = load_trained_cnn(checkpoint_path, device)
    
    # Load some test data
    test_loader = get_celeba_dataloader(split='test', batch_size=8, download=False)
    images, true_labels = next(iter(test_loader))
    
    # Predict attributes
    predictions = predict_attributes(model, images, device)
    
    # Get attribute names
    attr_names = get_attribute_names()
    
    # Display results for first image
    print("\nPredicted attributes for first image:")
    pred_attrs = predictions[0].cpu().numpy()
    true_attrs = true_labels[0].numpy()
    
    print("\nAttribute | Predicted | Actual")
    print("-" * 40)
    for i, name in enumerate(attr_names):
        print(f"{name:20} | {'Yes' if pred_attrs[i] else 'No':3} | {'Yes' if true_attrs[i] else 'No':3}")


def example_vae_latent_traversal():
    """
    Example of VAE latent space traversal.
    """
    print("\n=== VAE Latent Traversal Example ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint_path = 'checkpoints/vae_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the VAE first using train_vae.py")
        return
    
    model = load_trained_vae(checkpoint_path, device)
    
    # Perform random latent traversal
    print("Performing latent space traversal...")
    latent_dim = model.latent_dim
    traversals, base_z = random_latent_traversal(
        model, 
        latent_dim, 
        num_dims=6, 
        num_steps=10, 
        device=device
    )
    
    # Visualize
    os.makedirs('results', exist_ok=True)
    save_path = 'results/latent_traversal.png'
    visualize_multiple_traversals(traversals, save_path=save_path)
    print(f"Saved traversal visualization to {save_path}")


def example_vae_interpolation():
    """
    Example of VAE interpolation between two images.
    """
    print("\n=== VAE Interpolation Example ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint_path = 'checkpoints/vae_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the VAE first using train_vae.py")
        return
    
    model = load_trained_vae(checkpoint_path, device)
    
    # Load some test data
    test_loader = get_celeba_dataloader(split='test', batch_size=8, download=False)
    images, _ = next(iter(test_loader))
    
    # Interpolate between first two images
    print("Interpolating between two images...")
    interpolated = encode_and_interpolate(
        model, 
        images[0], 
        images[1], 
        num_steps=10, 
        device=device
    )
    
    # Visualize
    os.makedirs('results', exist_ok=True)
    save_path = 'results/interpolation.png'
    visualize_interpolation(interpolated, save_path=save_path)
    print(f"Saved interpolation visualization to {save_path}")


def example_vae_generation():
    """
    Example of generating random samples from VAE.
    """
    print("\n=== VAE Random Generation Example ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint_path = 'checkpoints/vae_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Please train the VAE first using train_vae.py")
        return
    
    model = load_trained_vae(checkpoint_path, device)
    
    # Generate random samples
    print("Generating random samples...")
    num_samples = 16
    samples = model.sample(num_samples, device=device)
    
    # Visualize
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_samples):
        row, col = i // 4, i % 4
        img = samples[i].cpu().detach()
        img = (img + 1) / 2  # Denormalize
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    save_path = 'results/random_samples.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved random samples to {save_path}")


if __name__ == '__main__':
    print("CelebA CNN and VAE Examples")
    print("=" * 50)
    
    # Run examples
    example_cnn_prediction()
    example_vae_latent_traversal()
    example_vae_interpolation()
    example_vae_generation()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
