import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import logging

class BaseRoofDataset(Dataset):
    """Base class for all roof datasets."""
    
    def __init__(self, data_dir, split='train', transform=None, max_samples=None):
        """
        Initialize the base dataset.
        
        Args:
            data_dir (str): Path to the dataset
            split (str): Either 'train', 'val', or 'test'
            transform: Optional transforms to apply
            max_samples (int): Maximum number of samples to load (for testing)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
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
        self.image_size = 512  # Standard size for all datasets
        
        # Each subclass must implement _load_dataset
        self._load_dataset()
        
        # Limit number of samples if specified
        if self.max_samples is not None and len(self.image_files) > self.max_samples:
            logging.info(f"Limiting dataset to {self.max_samples} samples (from {len(self.image_files)})")
            self.image_files = random.sample(self.image_files, self.max_samples)
        
        logging.info(f"Loaded {len(self.image_files)} images from {self.__class__.__name__} ({split} split)")
    
    def _load_dataset(self):
        """Load dataset - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_dataset")
    
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
    
    def preprocess_image(self, image):
        """Common image preprocessing."""
        if image is None:
            raise RuntimeError("Failed to load image")
        
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
    
    def preprocess_mask(self, mask):
        """Common mask preprocessing."""
        if mask is None:
            raise RuntimeError("Failed to load mask")
        
        # Resize if needed
        if mask.shape[:2] != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor
        mask = torch.from_numpy(mask).long()
        
        return mask
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Load image and mask - implemented by subclasses
            img_path = self.image_files[idx]
            image = self.load_image(img_path)
            mask = self.load_mask(img_path)
            
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
                'dataset_type': self.__class__.__name__
            }
            
            return image, features
            
        except Exception as e:
            logging.error(f"Error loading sample {img_path}: {str(e)}")
            raise