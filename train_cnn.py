"""
Training script for the CNN classifier on CelebA dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from cnn_classifier import CelebACNN
from data_utils import get_celeba_dataloader, get_attribute_names


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: CNN model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: CNN model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average loss and accuracy for validation set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device).float()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_cnn(num_epochs=20, batch_size=64, learning_rate=0.001, 
              image_size=64, device='cuda', save_dir='checkpoints'):
    """
    Train the CNN classifier on CelebA dataset.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        image_size: Size of input images
        device: Device to train on
        save_dir: Directory to save model checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
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
    model = CelebACNN(num_attributes=40, image_size=image_size).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'cnn_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, save_path)
            print(f"Saved best model to {save_path}")
    
    print("\nTraining completed!")
    return model


if __name__ == '__main__':
    # Train the model
    model = train_cnn(
        num_epochs=20,
        batch_size=64,
        learning_rate=0.001,
        image_size=64,
        device='cuda'
    )
