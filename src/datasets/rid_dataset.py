import cv2
import logging
from pathlib import Path
import torch
from .base_dataset import BaseRoofDataset

class RIDDataset(BaseRoofDataset):
    """RID (Roof Instance Dataset) implementation."""
    
    def _load_dataset(self):
        """Load RID dataset with correct folder structure."""
        # Set up paths for RID dataset subfolders
        split_dir = self.data_dir / 'filenames_train_val_test_split'
        self.img_dir = self.data_dir / 'images_roof_centered_geotiff'
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
                    names = [line.strip().rstrip('\r') for line in f.readlines()]
                    # Remove .png extension from filenames
                    names = [name[:-4] if name.endswith('.png') else name for name in names]
                    logging.info(f"Found {len(names)} filenames in {split_file}")
                    logging.info(f"Example filenames: {names[:5]}")
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
            
            # List some example filenames
            logging.error("Example filenames from split files:")
            example_names = list(image_names)[:5]
            for name in example_names:
                logging.error(f"  {name}")
            
            raise RuntimeError("No valid image/mask pairs found. Check paths and file structure.")
    
    def load_image(self, img_path):
        """Load and preprocess RID image."""
        image = cv2.imread(str(img_path))
        return self.preprocess_image(image)
    
    def load_mask(self, img_path):
        """Load and process RID segmentation mask."""
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = self.preprocess_mask(mask)
        
        # Convert to binary mask (0: background, 1: roof)
        mask = (mask > 0).long()
        
        return mask