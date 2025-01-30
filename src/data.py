"""
Data loading and preprocessing utilities for roof feature detection.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RoofDataset(Dataset):
    """
    Dataset class for loading and preprocessing roof images and their annotations.
    """
    
    def __init__(self, image_dir, mask_dir=None, transform=None, target_size=(1024, 1024)):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing input images
            mask_dir (str, optional): Directory containing annotation masks
            transform (callable, optional): Optional transform to be applied on images
            target_size (tuple): Target size for resizing images (default: 1024x1024)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of image files
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tif'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        # Load mask if available
        if self.mask_dir:
            mask_name = img_name.replace('.jpg', '_mask.png').replace('.tif', '_mask.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                mask = torch.from_numpy(mask).long()
            else:
                mask = torch.zeros(self.target_size).long()
            
            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            return image, mask
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image

def get_data_transforms():
    """
    Get default data augmentation transforms.
    
    Returns:
        dict: Dictionary containing train and validation transforms
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }

def create_dataloaders(image_dir, mask_dir=None, batch_size=8, num_workers=4):
    """
    Create DataLoader instances for training and validation.
    
    Args:
        image_dir (str): Directory containing input images
        mask_dir (str, optional): Directory containing annotation masks
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader) if mask_dir is provided,
               otherwise returns a single dataloader
    """
    transforms = get_data_transforms()
    
    if mask_dir:
        # Split into train/val
        dataset = RoofDataset(image_dir, mask_dir, transform=None)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Apply appropriate transforms
        train_dataset.transform = transforms['train']
        val_dataset.transform = transforms['val']
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    else:
        dataset = RoofDataset(image_dir, transform=transforms['val'])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
