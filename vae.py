"""
Variational Autoencoder (VAE) for CelebA dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Convolutional Variational Autoencoder for image generation.
    
    Architecture:
    - Encoder: Convolutional layers that compress input to latent representation
    - Latent space: Mean and log-variance vectors for sampling
    - Decoder: Transposed convolutional layers that reconstruct images
    """
    
    def __init__(self, latent_dim=128, image_size=64):
        """
        Initialize the VAE.
        
        Args:
            latent_dim: Dimensionality of the latent space
            image_size: Size of input/output images
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 256 x 4 x 4
        )
        
        # Calculate flattened size
        self.flat_size = 256 * 4 * 4
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 32 x 32
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 3 x 64 x 64
        )
        
    def encode(self, x):
        """
        Encode input images to latent space parameters.
        
        Args:
            x: Input tensor of shape (batch_size, 3, image_size, image_size)
            
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        h = self.encoder(x)
        h = h.view(-1, self.flat_size)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            
        Returns:
            Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vectors to images.
        
        Args:
            z: Latent vectors of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed images
        """
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            x_recon: Reconstructed images
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device='cpu'):
        """
        Generate random samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        x_recon: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log-variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
        
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div
