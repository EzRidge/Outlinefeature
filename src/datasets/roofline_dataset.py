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
            # Open the file to get dataset size
            with h5py.File(mat_file, 'r') as f:
                total_samples = f['imdb/images'].shape[0]
                split_idx = int(total_samples * 0.8)  # 80/20 split
                
                # Store file path and indices
                self.mat_file = str(mat_file)
                if self.split == 'train':
                    self.indices = list(range(split_idx))
                else:  # val split
                    self.indices = list(range(split_idx, total_samples))
                
                logging.info(f"Loaded {len(self.indices)} samples for {self.split} split")
                
                # Create list of indices as image_files
                self.image_files = self.indices
                
        except Exception as e:
            logging.error(f"Error loading HDF5 .mat file: {str(e)}")
            raise RuntimeError(f"Failed to load Roofline dataset: {str(e)}")
    
    def load_image(self, idx):
        """Load and preprocess Roofline image."""
        with h5py.File(self.mat_file, 'r') as f:
            # Get image from file (already in CHW format)
            image = f['imdb/images'][self.indices[idx]]
            
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
        with h5py.File(self.mat_file, 'r') as f:
            # Get elements from file (3 channels for different line types)
            elements = f['imdb/elements'][self.indices[idx]]
            
            # Convert to proper format
            if elements.dtype != np.uint8:
                elements = (elements * 255).astype(np.uint8)
            
            # Resize if needed
            if elements.shape[-2:] != (self.image_size, self.image_size):
                # Transpose to HWC for cv2, then back to CHW
                elements = elements.transpose(1, 2, 0)
                elements = cv2.resize(elements, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                elements = elements.transpose(2, 0, 1)
            
            # Convert 3-channel elements to single channel with class indices
            # Channel 0: Ridge lines (main peaks)
            # Channel 1: Hip lines (diagonal edges)
            # Channel 2: Valley lines (internal corners)
            final_mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            final_mask[elements[0] > 0.5] = self.class_mapping['ridge']
            final_mask[elements[1] > 0.5] = self.class_mapping['valley']
            final_mask[elements[2] > 0.5] = self.class_mapping['eave']
            
            # Add roof class where any line exists
            final_mask[elements.max(axis=0) > 0.5] = self.class_mapping['roof']
            
            return torch.from_numpy(final_mask)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Load image and mask
            img_path = self.image_files[idx]
            image = self.load_image(idx)
            mask = self.load_mask(idx)
            
            # Create line detection masks
            lines = self.create_line_mask(mask)
            
            # Load depth map
            with h5py.File(self.mat_file, 'r') as f:
                depth = f['imdb/depths'][self.indices[idx]]
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