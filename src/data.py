import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import os
import json
from PIL import Image
import scipy.io as sio
import h5py
import logging

class RoofDataset(Dataset):
    """Base dataset class for roof model training."""
    
    def __init__(self, data_dir, dataset_type, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the dataset
            dataset_type (str): Type of dataset ('rid', 'roofline', or 'airs')
            split (str): Either 'train', 'val', or 'test'
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
        """Load RID dataset with correct folder structure."""
        # Set up paths for RID dataset subfolders
        split_dir = self.data_dir / 'filenames_train_val_test_split'
        self.img_dir = self.data_dir / 'images_annotation_experiment_geotiff'
        self.mask_dir = self.data_dir / 'masks_segments_reviewed'
        
        logging.info(f"Loading RID dataset from:")
        logging.info(f"- Base directory: {self.data_dir}")
        logging.info(f"- Split directory: {split_dir}")
        logging.info(f"- Image directory: {self.img_dir}")
        logging.info(f"- Mask directory: {self.mask_dir}")
        
        # List all directories to verify structure
        logging.info("Available directories:")
        for item in self.data_dir.iterdir():
            if item.is_dir():
                logging.info(f"- {item.name}/")
        
        # Find split files for current split
        split_pattern = f"{self.split}_filenames_*_rev.txt"
        split_files = list(split_dir.glob(split_pattern))
        logging.info(f"Found {len(split_files)} split files matching {split_pattern}")
        
        if not split_files:
            raise RuntimeError(f"No split files found matching {split_pattern}")
        
        # Read filenames for current split
        image_names = set()
        for split_file in split_files:
            logging.info(f"Reading split file: {split_file}")
            try:
                with open(split_file, 'r') as f:
                    # Read lines and clean up filenames
                    names = [line.strip() for line in f.readlines()]
                    # Remove any file extensions
                    names = [os.path.splitext(name)[0] for name in names]
                    # Remove any remaining extensions (in case of double extensions)
                    names = [os.path.splitext(name)[0] for name in names]
                    logging.info(f"Found {len(names)} filenames in {split_file}")
                    image_names.update(names)
            except Exception as e:
                logging.error(f"Error reading {split_file}: {str(e)}")
        
        # Get image paths and verify files exist
        self.image_files = []
        for name in image_names:
            # Try both with and without leading zeros
            possible_names = [
                name,  # Original name
                name.zfill(4),  # With leading zeros
                name.lstrip('0')  # Without leading zeros
            ]
            
            found = False
            for possible_name in possible_names:
                img_path = self.img_dir / f"{possible_name}.tif"  # Images are .tif
                mask_path = self.mask_dir / f"{possible_name}.png"  # Masks are .png
                
                if img_path.exists() and mask_path.exists():
                    self.image_files.append(img_path)
                    found = True
                    logging.info(f"Found image/mask pair: {img_path.name}")
                    break
            
            if not found:
                logging.warning(f"Could not find image/mask pair for {name}")
                logging.warning(f"Tried:")
                for possible_name in possible_names:
                    logging.warning(f"- Image: {self.img_dir / f'{possible_name}.tif'}")
                    logging.warning(f"- Mask: {self.mask_dir / f'{possible_name}.png'}")
        
        logging.info(f"Successfully loaded {len(self.image_files)} image/mask pairs")
        if len(self.image_files) == 0:
            logging.error("No valid image/mask pairs found!")
            logging.error("Directory contents:")
            if self.img_dir.exists():
                logging.error(f"Image directory ({self.img_dir}):")
                for f in self.img_dir.glob('*'):
                    logging.error(f"  {f.name}")
            else:
                logging.error(f"Image directory does not exist: {self.img_dir}")
            
            if self.mask_dir.exists():
                logging.error(f"Mask directory ({self.mask_dir}):")
                for f in self.mask_dir.glob('*'):
                    logging.error(f"  {f.name}")
            else:
                logging.error(f"Mask directory does not exist: {self.mask_dir}")
        
    def _load_roofline_dataset(self):
        """Load Roofline dataset from .mat file."""
        mat_file = next(self.data_dir.glob('*.mat'))
        logging.info(f"Loading Roofline dataset from: {mat_file}")
        
        try:
            # Try loading as v7.3 .mat file
            with h5py.File(mat_file, 'r') as f:
                total_samples = len(f['images'])
                split_idx = int(total_samples * 0.8)  # 80/20 split
                
                if self.split == 'train':
                    self.images = f['images'][:split_idx]
                    self.masks = f['masks'][:split_idx]
                else:
                    self.images = f['images'][split_idx:]
                    self.masks = f['masks'][split_idx:]
                    
                logging.info(f"Loaded {len(self.images)} samples from HDF5 .mat file")
        except:
            # Fall back to older .mat format
            logging.info("Falling back to legacy .mat format")
            data = sio.loadmat(mat_file)
            total_samples = len(data['images'])
            split_idx = int(total_samples * 0.8)  # 80/20 split
            
            if self.split == 'train':
                self.images = data['images'][:split_idx]
                self.masks = data['masks'][:split_idx]
            else:
                self.images = data['images'][split_idx:]
                self.masks = data['masks'][split_idx:]
            
            logging.info(f"Loaded {len(self.images)} samples from legacy .mat file")
        
        self.image_files = list(range(len(self.images)))
    
    def _load_airs_dataset(self):
        """Load AIRS dataset with existing splits."""
        # AIRS dataset has a specific directory structure
        image_dir = self.data_dir / self.split / 'image'
        mask_dir = self.data_dir / self.split / 'label'
        
        logging.info(f"Loading AIRS dataset from:")
        logging.info(f"- Image directory: {image_dir}")
        logging.info(f"- Mask directory: {mask_dir}")
        
        # Get all image files
        self.image_files = sorted(list(image_dir.glob('*.tif')))
        self.mask_dir = mask_dir
        
        logging.info(f"Found {len(self.image_files)} images")
    
    def load_image(self, img_path):
        """Load and preprocess image."""
        try:
            if isinstance(img_path, Path):
                # Handle .tif files for RID and AIRS
                if str(img_path).endswith('.tif'):
                    image = cv2.imread(str(img_path))
                else:
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
            
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            raise
    
    def load_mask(self, img_path):
        """Load and process segmentation mask."""
        try:
            if isinstance(img_path, Path):
                if self.dataset_type == 'airs':
                    # AIRS masks are in label directory
                    mask_path = self.mask_dir / f"{img_path.stem}_label.png"
                elif self.dataset_type == 'rid':
                    # RID masks are in masks_segments_reviewed directory
                    mask_path = self.mask_dir / f"{img_path.stem}.png"
                else:
                    mask_path = img_path.with_suffix('.png')
                    
                logging.debug(f"Loading mask from: {mask_path}")
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            else:  # For Roofline dataset stored in memory
                mask = self.masks[img_path]
                
            if mask is None:
                raise RuntimeError(f"Failed to load mask: {mask_path if isinstance(img_path, Path) else img_path}")
            
            # Resize if needed
            if mask.shape[:2] != (self.image_size, self.image_size):
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensor
            mask = torch.from_numpy(mask).long()
            
            return mask
            
        except Exception as e:
            logging.error(f"Error loading mask for {img_path}: {str(e)}")
            raise
    
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
                'dataset_type': self.dataset_type  # Include dataset type for specialized loss
            }
            
            return image, features
            
        except Exception as e:
            logging.error(f"Error loading sample {img_path}: {str(e)}")
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
    logging.info(f"Creating dataloaders for {dataset_type} dataset")
    logging.info(f"Dataset path: {dataset_path}")
    
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
    
    logging.info(f"Created dataloaders:")
    logging.info(f"- Training samples: {len(train_dataset)}")
    logging.info(f"- Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader
