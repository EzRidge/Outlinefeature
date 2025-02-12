import cv2
import logging
import h5py
import numpy as np
import torch
from pathlib import Path
from .base_dataset import BaseRoofDataset

class RooflineDataset(BaseRoofDataset):
    """Roofline dataset implementation."""
    
    def _load_dataset(self):
        """Load Roofline dataset from .mat file."""
        # Find .mat file
        mat_file = next(self.data_dir.glob('*.mat'))
        logging.info(f"Loading Roofline dataset from: {mat_file}")
        
        try:
            # Load v7.3 .mat file using h5py
            with h5py.File(mat_file, 'r') as f:
                # Get dataset references
                total_samples = f['imdb/images'].shape[0]
                split_idx = int(total_samples * 0.8)  # 80/20 split
                
                # Load data into memory based on split
                if self.split == 'train':
                    self.images = f['imdb/images'][:split_idx]
                    self.elements = f['imdb/elements'][:split_idx]
                    self.depths = f['imdb/depths'][:split_idx]
                else:  # val split
                    self.images = f['imdb/images'][split_idx:]
                    self.elements = f['imdb/elements'][split_idx:]
                    self.depths = f['imdb/depths'][split_idx:]
                
                # Create list of indices as image_files
                self.image_files = list(range(len(self.images)))
                
                logging.info(f"Loaded {len(self.image_files)} samples for {self.split} split")
                
        except Exception as e:
            logging.error(f"Error loading HDF5 .mat file: {str(e)}")
            raise RuntimeError(f"Failed to load Roofline dataset: {str(e)}")
    
    def load_image(self, idx):
        """Load and preprocess Roofline image."""
        # Get image from memory (already in CHW format)
        image = self.images[idx]
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Resize if needed (from 224x224 to 512x512)
        if image.shape[-2:] != (self.image_size, self.image_size):
            # Transpose to HWC for cv2, then back to CHW
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image)
    
    def load_mask(self, idx):
        """Load and process Roofline segmentation mask."""
        # Get mask from memory (3 channels for different elements)
        mask = self.elements[idx]
        
        # Convert to proper format
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Resize if needed
        if mask.shape[-2:] != (self.image_size, self.image_size):
            # Transpose to HWC for cv2, then back to CHW
            mask = mask.transpose(1, 2, 0)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = mask.transpose(2, 0, 1)
        
        # Convert 3-channel mask to single channel with class indices
        # Channel 0: Roof segments
        # Channel 1: Ridge lines
        # Channel 2: Valley lines
        final_mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
        final_mask[mask[0] > 0.5] = self.class_mapping['roof']
        final_mask[mask[1] > 0.5] = self.class_mapping['ridge']
        final_mask[mask[2] > 0.5] = self.class_mapping['valley']
        
        return torch.from_numpy(final_mask)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Load image and mask
            img_path = self.image_files[idx]
            image = self.load_image(img_path)
            mask = self.load_mask(img_path)
            
            # Create line detection masks
            lines = self.create_line_mask(mask)
            
            # Load depth map
            depth = self.depths[img_path]
            if depth.shape != (self.image_size, self.image_size):
                depth = cv2.resize(depth, (self.image_size, self.image_size))
            depth = torch.from_numpy(depth.astype(np.float32))
            
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