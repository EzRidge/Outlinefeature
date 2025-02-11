import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import json
from PIL import Image

class UnifiedRoofDataset(Dataset):
    """Dataset for unified roof model training with multiple datasets."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the unified dataset.
        
        Args:
            data_dir (str): Path to the processed unified dataset
            split (str): Either 'train' or 'val'
            transform: Optional transforms to apply
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Set up paths
        self.img_dir = self.data_dir / split / 'images'
        self.mask_dir = self.data_dir / split / 'masks'
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.class_mapping = self.metadata['class_mapping']
        self.num_classes = self.metadata['num_classes']
        self.image_size = self.metadata['image_size']
        
        # Get all image files
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')))
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.img_dir}")
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def load_image(self, img_path):
        """Load and preprocess image."""
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange channels
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def load_mask(self, mask_path):
        """Load and process segmentation mask."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        
        # Convert to tensor
        mask = torch.from_numpy(mask).long()
        
        return mask
    
    def create_line_mask(self, mask):
        """Create line detection masks from segmentation mask."""
        # Initialize line masks for ridge, valley, eave, and outline
        h, w = mask.shape
        lines = torch.zeros(4, h, w)
        
        # Extract lines based on class values
        ridge_mask = (mask == self.class_mapping['ridge']).float()
        valley_mask = (mask == self.class_mapping['valley']).float()
        eave_mask = (mask == self.class_mapping['eave']).float()
        
        # Create outline from roof class
        roof_mask = (mask == self.class_mapping['roof']).float()
        kernel = np.ones((3, 3), np.uint8)
        roof_dilated = cv2.dilate(roof_mask.numpy(), kernel, iterations=1)
        outline_mask = torch.from_numpy(roof_dilated - roof_mask.numpy()).float()
        
        lines[0] = ridge_mask    # Ridge lines
        lines[1] = valley_mask   # Valley lines
        lines[2] = eave_mask     # Eave lines
        lines[3] = outline_mask  # Roof outline
        
        return lines
    
    def estimate_depth(self, mask):
        """
        Estimate pseudo-depth from segmentation mask.
        This is a simple heuristic until we have real depth data.
        """
        h, w = mask.shape
        depth = torch.zeros(h, w)
        
        # Use height information from different roof elements
        depth[mask == self.class_mapping['ridge']] = 1.0    # Highest points
        depth[mask == self.class_mapping['valley']] = 0.5   # Mid-level
        depth[mask == self.class_mapping['eave']] = 0.2     # Lower points
        depth[mask == self.class_mapping['roof']] = 0.3     # Base roof level
        
        # Smooth the depth map
        depth = cv2.GaussianBlur(depth.numpy(), (5, 5), 0)
        depth = torch.from_numpy(depth).float()
        
        return depth
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"
        
        try:
            # Load image and mask
            image = self.load_image(img_path)
            mask = self.load_mask(mask_path)
            
            # Create line detection masks
            lines = self.create_line_mask(mask)
            
            # Estimate depth
            depth = self.estimate_depth(mask)
            
            # Apply transforms if any
            if self.transform is not None:
                image = self.transform(image)
            
            # Prepare feature dictionary
            features = {
                'segments': mask,
                'lines': lines,
                'depth': depth
            }
            
            return image, features
            
        except Exception as e:
            print(f"Error loading sample {img_path}: {str(e)}")
            # Return a dummy sample in case of error
            return self.__getitem__((idx + 1) % len(self))

def create_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir (str): Path to processed dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
        
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Create datasets
    train_dataset = UnifiedRoofDataset(data_dir, split='train')
    val_dataset = UnifiedRoofDataset(data_dir, split='val')
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
