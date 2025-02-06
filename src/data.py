import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image
import rasterio

class RoofDataset(Dataset):
    """Dataset for hybrid roof model combining RID and Roofline data."""
    
    def __init__(self, rid_indices, roofline_indices):
        """
        Initialize the dataset.
        
        Args:
            rid_indices: List of RID dataset indices
            roofline_indices: List of Roofline dataset indices
        """
        super().__init__()
        self.rid_indices = rid_indices
        self.roofline_indices = roofline_indices
        
        # Class mapping based on paper insights
        # Keep high-agreement classes separate, merge others
        self.CLASS_MAPPING = {
            0: 0,  # Background -> Background
            1: 1,  # PV module -> PV module (high agreement, 0.68 IoU)
            2: 2,  # Dormer -> Dormer (high agreement, 0.70 IoU)
            3: 3,  # Window -> Window
            4: 4,  # Ladder -> Ladder
            5: 5,  # Chimney -> Chimney
            6: 6,  # Shadow -> Shadow
            7: 7,  # Tree -> Tree
            8: 8   # Unknown -> Unknown
        }
        
        # Base paths
        self.rid_base = os.path.join("Reference Materials", "data", "RID", "m1655470", "RID_dataset")
        self.roofline_base = os.path.join("Reference Materials", "data", "Roofline-Extraction")
        
        # RID paths
        self.rid_images = os.path.join(self.rid_base, "images_roof_centered_geotiff")
        self.rid_segments = os.path.join(self.rid_base, "masks_segments_reviewed")
        self.rid_superstructures = os.path.join(self.rid_base, "masks_superstructures_reviewed")
        
        # Verify paths exist
        required_paths = [
            self.rid_base, self.roofline_base,
            self.rid_images, self.rid_segments, self.rid_superstructures
        ]
        for path in required_paths:
            if not os.path.exists(path):
                raise RuntimeError(f"Required path does not exist: {path}")
    
    def _remap_classes(self, mask):
        """Remap class labels according to CLASS_MAPPING."""
        remapped = np.zeros_like(mask)
        for old_val, new_val in self.CLASS_MAPPING.items():
            remapped[mask == old_val] = new_val
        return remapped
    
    def __len__(self):
        return len(self.rid_indices)
    
    def load_rid_sample(self, idx):
        """Load a sample from the RID dataset."""
        sample_id = self.rid_indices[idx]
        
        # Load image
        image_path = os.path.join(self.rid_images, f"{sample_id}.tif")
        print(f"Loading image: {image_path}")
        if not os.path.exists(image_path):
            error_msg = f"File not found: {image_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load and process image
            with rasterio.open(image_path) as src:
                image_data = src.read()
                print(f"Successfully read image data for sample {sample_id}")
                print(f"Image shape: {image_data.shape}")
                print(f"Image dtype: {image_data.dtype}")
                print(f"Image value range: [{image_data.min()}, {image_data.max()}]")
                
                # Convert from CHW to HWC for OpenCV processing
                image = image_data.transpose(1, 2, 0)
                
                # Resize with INTER_AREA for downsampling to prevent aliasing
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Convert to tensor and normalize
                image = torch.from_numpy(image).float() / 255.0
                
                # Convert back to CHW for PyTorch
                image = image.permute(2, 0, 1)
                
                print(f"Successfully processed image for sample {sample_id}")
            
            # Load segments
            segments_path = os.path.join(self.rid_segments, f"{sample_id}.png")
            with rasterio.open(segments_path) as src:
                segments = src.read(1)  # Read first band
                # Use nearest neighbor for label maps to avoid creating invalid classes
                segments = cv2.resize(segments, (224, 224), interpolation=cv2.INTER_NEAREST)
                segments = self._remap_classes(segments)
                segments = torch.from_numpy(segments).long()
                print(f"Successfully loaded segments for sample {sample_id}")
            
            # Load superstructures
            superstructures_path = os.path.join(self.rid_superstructures, f"{sample_id}.png")
            with rasterio.open(superstructures_path) as src:
                superstructures = src.read(1)  # Read first band
                # Use nearest neighbor for label maps to avoid creating invalid classes
                superstructures = cv2.resize(superstructures, (224, 224), interpolation=cv2.INTER_NEAREST)
                superstructures = self._remap_classes(superstructures)
                superstructures = torch.from_numpy(superstructures).float()
                print(f"Successfully loaded superstructures for sample {sample_id}")
            
            return {
                'image': image,
                'segments': segments,
                'superstructures': superstructures
            }
            
        except Exception as e:
            error_msg = f"Error processing sample {sample_id}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def load_roofline_sample(self, idx):
        """Load a sample from the Roofline dataset."""
        sample_id = self.roofline_indices[idx]
        
        # TODO: Implement actual Roofline data loading
        # For now, return dummy tensors
        h, w = 224, 224
        
        depth = torch.randn(h, w)  # Dummy depth map
        
        # Create combined lines tensor [3, H, W] for ridge, hip, valley
        lines = torch.zeros(3, h, w)  # [ridge, hip, valley]
        
        return {
            'depth': depth,
            'lines': lines
        }
    
    def __getitem__(self, idx):
        """Get a single item combining both datasets."""
        # Load samples from both datasets
        rid_sample = self.load_rid_sample(idx)
        roofline_sample = self.load_roofline_sample(idx)
        
        # Combine features
        features = {
            'image': rid_sample['image'],
            'segments': rid_sample['segments'],
            'superstructures': rid_sample['superstructures'],
            'depth': roofline_sample['depth'],
            'lines': roofline_sample['lines']  # Now a single tensor [3, H, W]
        }
        
        return features['image'], features
