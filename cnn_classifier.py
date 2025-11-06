"""
CNN Classifier for CelebA attribute prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CelebACNN(nn.Module):
    """
    Convolutional Neural Network for CelebA attribute classification.
    
    Architecture:
    - Multiple convolutional layers with batch normalization and pooling
    - Fully connected layers for classification
    - Multi-label output (40 attributes)
    """
    
    def __init__(self, num_attributes=40, image_size=64):
        """
        Initialize the CNN.
        
        Args:
            num_attributes: Number of attributes to predict (40 for CelebA)
            image_size: Size of input images
        """
        super(CelebACNN, self).__init__()
        
        self.num_attributes = num_attributes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size after convolutions
        # image_size -> image_size/2 -> image_size/4 -> image_size/8 -> image_size/16 -> image_size/32
        self.flat_size = 512 * (image_size // 32) * (image_size // 32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_attributes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, image_size, image_size)
            
        Returns:
            Output tensor of shape (batch_size, num_attributes)
        """
        # Convolutional layers with ReLU, batch norm, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        """
        Get binary predictions from model output.
        
        Args:
            x: Input tensor
            
        Returns:
            Binary predictions (0 or 1) for each attribute
        """
        logits = self.forward(x)
        return torch.sigmoid(logits) > 0.5
