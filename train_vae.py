"""
Training script for the Variational Autoencoder on CelebA dataset.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

from vae import VAE, vae_loss
from data_utils import get_celeba_dataloader


def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    """
    Train the VAE for one epoch.
    
    Args:
        model: VAE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        beta: Weight for KL divergence (beta-VAE)
        
    Returns:
        Average total loss, reconstruction loss, and KL divergence
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_div_sum = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, _ in pbar:
        images = images.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, mu, logvar = model(images)
        
        # Calculate loss
        total_loss, recon_loss, kl_div = vae_loss(x_recon, images, mu, logvar, beta)
        
        # Normalize by batch size
        batch_size = images.size(0)
        total_loss = total_loss / batch_size
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item() / batch_size
        kl_div_sum += kl_div.item() / batch_size
        
        pbar.set_postfix({
            'loss': total_loss.item(),
            'recon': recon_loss.item() / batch_size,
            'kl': kl_div.item() / batch_size
        })
    
    n_batches = len(dataloader)
    return (total_loss_sum / n_batches, 
            recon_loss_sum / n_batches, 
            kl_div_sum / n_batches)


def validate(model, dataloader, device, beta=1.0):
    """
    Validate the VAE.
    
    Args:
        model: VAE model
        dataloader: Validation data loader
        device: Device to validate on
        beta: Weight for KL divergence
        
    Returns:
        Average total loss, reconstruction loss, and KL divergence
    """
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_div_sum = 0.0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(images)
            
            # Calculate loss
            total_loss, recon_loss, kl_div = vae_loss(x_recon, images, mu, logvar, beta)
            
            # Normalize by batch size
            batch_size = images.size(0)
            total_loss_sum += total_loss.item() / batch_size
            recon_loss_sum += recon_loss.item() / batch_size
            kl_div_sum += kl_div.item() / batch_size
    
    n_batches = len(dataloader)
    return (total_loss_sum / n_batches,
            recon_loss_sum / n_batches,
            kl_div_sum / n_batches)


def save_reconstruction_samples(model, dataloader, device, save_path, num_samples=8):
    """
    Save reconstruction samples for visualization.
    
    Args:
        model: VAE model
        dataloader: Data loader
        device: Device
        save_path: Path to save the image
        num_samples: Number of samples to show
    """
    model.eval()
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        x_recon, _, _ = model(images)
    
    # Denormalize images
    images = (images + 1) / 2
    x_recon = (x_recon + 1) / 2
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        # Reconstructed image
        recon = x_recon[i].cpu().permute(1, 2, 0).numpy()
        recon = np.clip(recon, 0, 1)
        axes[1, i].imshow(recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def train_vae(num_epochs=50, batch_size=64, learning_rate=0.001, 
              latent_dim=128, image_size=64, beta=1.0, 
              device='cuda', save_dir='checkpoints'):
    """
    Train the VAE on CelebA dataset.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        latent_dim: Dimensionality of latent space
        image_size: Size of input images
        beta: Weight for KL divergence (beta-VAE)
        device: Device to train on
        save_dir: Directory to save model checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader = get_celeba_dataloader(
        split='train', 
        batch_size=batch_size, 
        image_size=image_size,
        download=True
    )
    
    val_loader = get_celeba_dataloader(
        split='valid', 
        batch_size=batch_size, 
        image_size=image_size,
        download=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = VAE(latent_dim=latent_dim, image_size=image_size).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    print(f"\nTraining VAE for {num_epochs} epochs (beta={beta})...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, beta
        )
        print(f"Training - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        
        # Validate
        val_loss, val_recon, val_kl = validate(model, val_loader, device, beta)
        print(f"Validation - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save reconstruction samples
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join('results', f'reconstruction_epoch_{epoch+1}.png')
            save_reconstruction_samples(model, val_loader, device, save_path)
            print(f"Saved reconstruction samples to {save_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'vae_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon': val_recon,
                'val_kl': val_kl,
                'latent_dim': latent_dim,
                'image_size': image_size
            }, save_path)
            print(f"Saved best model to {save_path}")
    
    print("\nTraining completed!")
    return model


if __name__ == '__main__':
    # Train the model
    model = train_vae(
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        latent_dim=128,
        image_size=64,
        beta=1.0,
        device='cuda'
    )
