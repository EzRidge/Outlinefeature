"""
Dataset handling for roof feature detection.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
from .config import MODEL_CONFIG, DATA_DIR

class RoofDataset(Dataset):
    """Dataset for roof images and their feature annotations."""
    
    def __init__(self, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            transform: Optional transform to be applied to images
        """
        super().__init__()
        self.split = split
        self.transform = transform
        self.input_size = MODEL_CONFIG['input_size']
        
        # Set up paths
        self.data_dir = DATA_DIR / split
        self.image_dir = self.data_dir / 'images'
        self.annotation_dir = self.data_dir / 'annotations'
        
        # Load image paths and annotations
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        self.annotation_paths = sorted(list(self.annotation_dir.glob('*.json')))
        
        assert len(self.image_paths) == len(self.annotation_paths), \
            f"Number of images ({len(self.image_paths)}) and annotations ({len(self.annotation_paths)}) don't match"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation_path = self.annotation_paths[idx]
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        # Prepare feature maps
        h, w = self.input_size
        outline_map = np.zeros((h, w), dtype=np.float32)
        ridge_map = np.zeros((h, w), dtype=np.float32)
        hip_map = np.zeros((h, w), dtype=np.float32)
        valley_map = np.zeros((h, w), dtype=np.float32)
        angle_map = np.zeros((h, w), dtype=np.float32)
        
        # Draw features on maps
        self._draw_outline(outline_map, annotation['outline'])
        self._draw_lines(ridge_map, annotation['ridge_lines'])
        self._draw_lines(hip_map, annotation['hip_lines'])
        self._draw_lines(valley_map, annotation['valley_lines'])
        self._draw_angles(angle_map, annotation['angles'])
        
        # Resize image and maps
        image = cv2.resize(image, (w, h))
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        targets = {
            'outline': torch.from_numpy(outline_map)[None],  # Add channel dimension
            'ridge': torch.from_numpy(ridge_map)[None],
            'hip': torch.from_numpy(hip_map)[None],
            'valley': torch.from_numpy(valley_map)[None],
            'angles': torch.from_numpy(angle_map)[None]
        }
        
        return image, targets
    
    def _draw_outline(self, mask, points):
        """Draw roof outline on mask."""
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    
    def _draw_lines(self, mask, lines):
        """Draw lines (ridge, hip, valley) on mask."""
        for line in lines:
            pt1 = tuple(map(int, line[0]))
            pt2 = tuple(map(int, line[1]))
            cv2.line(mask, pt1, pt2, 1, thickness=2)
    
    def _draw_angles(self, mask, angles):
        """Draw angle predictions on mask."""
        for angle_data in angles:
            x, y = map(int, angle_data['position'])
            angle = angle_data['angle']
            mask[y, x] = angle

def create_dataloaders(batch_size, num_workers=4, pin_memory=True):
    """Create training and validation dataloaders."""
    train_dataset = RoofDataset(split='train')
    val_dataset = RoofDataset(split='val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
