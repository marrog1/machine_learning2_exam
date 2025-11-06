"""
Data utilities for loading and preprocessing the CelebA dataset.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_celeba_dataloader(root='./data', split='train', batch_size=64, 
                          image_size=64, num_workers=2, download=True):
    """
    Get a DataLoader for the CelebA dataset.
    
    Args:
        root: Root directory for the dataset
        split: Dataset split ('train', 'valid', 'test', or 'all')
        batch_size: Batch size for the DataLoader
        image_size: Size to resize images to
        num_workers: Number of worker processes for data loading
        download: Whether to download the dataset if not present
        
    Returns:
        DataLoader for the CelebA dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    dataset = datasets.CelebA(
        root=root,
        split=split,
        target_type='attr',
        transform=transform,
        download=download
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_attribute_names():
    """
    Returns the list of attribute names in CelebA dataset.
    """
    return [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young'
    ]
