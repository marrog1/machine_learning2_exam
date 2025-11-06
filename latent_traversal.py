"""
Latent space traversal utilities for exploring the VAE's learned representations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def linear_interpolation(vae, z1, z2, num_steps=10, device='cpu'):
    """
    Perform linear interpolation between two latent vectors.
    
    Args:
        vae: Trained VAE model
        z1: Starting latent vector
        z2: Ending latent vector
        num_steps: Number of interpolation steps
        device: Device to run on
        
    Returns:
        Interpolated images
    """
    vae.eval()
    with torch.no_grad():
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps).to(device)
        
        # Interpolate between z1 and z2
        z_interp = torch.stack([
            (1 - alpha) * z1 + alpha * z2 
            for alpha in alphas
        ])
        
        # Decode interpolated latent vectors
        images = vae.decode(z_interp)
        
    return images


def latent_traversal(vae, base_z, dim_idx, min_val=-3, max_val=3, num_steps=10, device='cpu'):
    """
    Traverse a single dimension of the latent space while keeping others fixed.
    
    Args:
        vae: Trained VAE model
        base_z: Base latent vector to modify
        dim_idx: Index of the dimension to traverse
        min_val: Minimum value for traversal
        max_val: Maximum value for traversal
        num_steps: Number of steps in the traversal
        device: Device to run on
        
    Returns:
        Images from traversal
    """
    vae.eval()
    with torch.no_grad():
        # Create values for traversal
        values = torch.linspace(min_val, max_val, num_steps).to(device)
        
        # Create modified latent vectors
        z_traversal = base_z.repeat(num_steps, 1)
        z_traversal[:, dim_idx] = values
        
        # Decode traversal
        images = vae.decode(z_traversal)
        
    return images


def random_latent_traversal(vae, latent_dim, num_dims=10, num_steps=10, device='cpu'):
    """
    Traverse multiple random dimensions of the latent space.
    
    Args:
        vae: Trained VAE model
        latent_dim: Dimensionality of latent space
        num_dims: Number of dimensions to traverse
        num_steps: Number of steps per dimension
        device: Device to run on
        
    Returns:
        Dictionary with dimension indices and corresponding images
    """
    vae.eval()
    
    # Sample a random base latent vector
    base_z = torch.randn(1, latent_dim).to(device)
    
    # Randomly select dimensions to traverse
    dim_indices = np.random.choice(latent_dim, size=num_dims, replace=False)
    
    traversals = {}
    for dim_idx in dim_indices:
        images = latent_traversal(vae, base_z, dim_idx, num_steps=num_steps, device=device)
        traversals[dim_idx] = images
        
    return traversals, base_z


def visualize_interpolation(images, save_path=None, title="Latent Interpolation"):
    """
    Visualize interpolation results.
    
    Args:
        images: Tensor of images
        save_path: Path to save the visualization
        title: Title for the plot
    """
    num_images = images.shape[0]
    
    fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2))
    
    for i in range(num_images):
        img = images[i].cpu().detach()
        img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        if num_images == 1:
            axes.imshow(img)
            axes.axis('off')
        else:
            axes[i].imshow(img)
            axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()


def visualize_traversal(images, dim_idx, save_path=None):
    """
    Visualize a single dimension traversal.
    
    Args:
        images: Tensor of images from traversal
        dim_idx: Index of the traversed dimension
        save_path: Path to save the visualization
    """
    visualize_interpolation(
        images, 
        save_path=save_path,
        title=f"Latent Dimension {dim_idx} Traversal"
    )


def visualize_multiple_traversals(traversals, save_path=None):
    """
    Visualize multiple dimension traversals in a grid.
    
    Args:
        traversals: Dictionary mapping dimension indices to image tensors
        save_path: Path to save the visualization
    """
    num_dims = len(traversals)
    dim_indices = list(traversals.keys())
    
    num_steps = traversals[dim_indices[0]].shape[0]
    
    fig = plt.figure(figsize=(2*num_steps, 2*num_dims))
    gs = GridSpec(num_dims, num_steps, figure=fig, hspace=0.3, wspace=0.05)
    
    for row, dim_idx in enumerate(dim_indices):
        images = traversals[dim_idx]
        
        for col in range(num_steps):
            ax = fig.add_subplot(gs[row, col])
            
            img = images[col].cpu().detach()
            img = (img + 1) / 2  # Denormalize
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(f'Dim {dim_idx}', fontsize=10)
    
    plt.suptitle("Latent Space Traversal", fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()


def encode_and_interpolate(vae, img1, img2, num_steps=10, device='cpu'):
    """
    Encode two images and interpolate between them in latent space.
    
    Args:
        vae: Trained VAE model
        img1: First image tensor
        img2: Second image tensor
        num_steps: Number of interpolation steps
        device: Device to run on
        
    Returns:
        Interpolated images
    """
    vae.eval()
    with torch.no_grad():
        # Encode images to latent space
        mu1, _ = vae.encode(img1.unsqueeze(0).to(device))
        mu2, _ = vae.encode(img2.unsqueeze(0).to(device))
        
        # Interpolate
        images = linear_interpolation(vae, mu1, mu2, num_steps, device)
        
    return images
