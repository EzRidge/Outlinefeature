import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import json
from PIL import Image
import scipy.io as sio

class RoofDataset(Dataset):
    """Base dataset class for roof model training."""
    
    def __init__(self, data_dir, dataset_type, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the dataset
            dataset_type (str): Type of dataset ('rid', 'roofline', or 'airs')
            split (str): Either 'train' or 'val'
            transform: Optional transforms to apply
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        
        # Unified class mapping for all datasets
        self.class_mapping = {
            'roof': 1,
            'ridge': 2,
            'valley': 3,
            'eave': 4,
            'dormer': 5,
            'chimney': 6,
            'window': 7,
            'pv_module': 8,
            'shadow': 9,
            'tree': 10,
            'unknown': 11
        }
        self.num_classes = len(self.class_mapping)
        self.image_size = 1024
        
        # Load dataset based on type
        if dataset_type == 'rid':
            self._load_rid_dataset()
        elif dataset_type == 'roofline':
            self._load_roofline_dataset()
        elif dataset_type == 'airs':
            self._load_airs_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        print(f"Loaded {len(self.image_files)} images from {dataset_type} dataset ({split} split)")
    
    def _load_rid_dataset(self):
        """Load RID dataset - focuses on general roof segmentation."""
        self.img_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')))
        
    def _load_roofline_dataset(self):
        """Load Roofline dataset - specializes in line detection."""
        mat_file = next(self.data_dir.glob('*.mat'))
        try:
            data = sio.loadmat(mat_file)
            # Split data for train/val
            total_samples = len(data['images'])
            split_idx = int(total_samples * 0.8)  # 80/20 split
            
            if self.split == 'train':
                self.images = data['images'][:split_idx]
                self.masks = data['masks'][:split_idx]
            else:
                self.images = data['images'][split_idx:]
                self.masks = data['masks'][split_idx:]
                
            self.image_files = list(range(len(self.images)))
        except Exception as e:
            raise RuntimeError(f"Failed to load Roofline .mat file: {mat_file}. Error: {str(e)}")
    
    def _load_airs_dataset(self):
        """Load AIRS dataset - strong on building footprints."""
        split_dir = self.data_dir / self.split
        self.img_dir = split_dir / 'images'
        self.mask_dir = split_dir / 'masks'
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')))
    
    def load_image(self, img_path):
        """Load and preprocess image."""
        if isinstance(img_path, Path):
            image = cv2.imread(str(img_path))
        else:  # For Roofline dataset stored in memory
            image = self.images[img_path]
            
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange channels
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def load_mask(self, mask_path):
        """Load and process segmentation mask."""
        if isinstance(mask_path, Path):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:  # For Roofline dataset stored in memory
            mask = self.masks[mask_path]
            
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        
        # Resize if needed
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor
        mask = torch.from_numpy(mask).long()
        
        return mask
    
    def create_line_mask(self, mask):
        """Create line detection masks from segmentation mask."""
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
        """Estimate pseudo-depth from segmentation mask."""
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
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Load image and mask
            if self.dataset_type == 'roofline':
                image = self.load_image(idx)
                mask = self.load_mask(idx)
            else:
                img_path = self.image_files[idx]
                mask_path = self.mask_dir / f"{img_path.stem}_mask.png"
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
                'depth': depth,
                'dataset_type': self.dataset_type  # Include dataset type for specialized loss
            }
            
            return image, features
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample in case of error
            return self.__getitem__((idx + 1) % len(self))

def create_dataloaders(dataset_path, dataset_type, batch_size=16, num_workers=4):
    """
    Create training and validation dataloaders for a specific dataset.
    
    Args:
        dataset_path (str): Path to dataset
        dataset_type (str): Type of dataset ('rid', 'roofline', or 'airs')
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
        
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Create datasets
    train_dataset = RoofDataset(dataset_path, dataset_type, split='train')
    val_dataset = RoofDataset(dataset_path, dataset_type, split='val')
    
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
