"""
Dataset preparation script that handles both RID and Roofline-Extraction datasets.
"""

import os
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import h5py
import scipy.io as sio
from .config import DATASET_CONFIG

class DatasetPreparator:
    def __init__(self, output_dir):
        """
        Initialize dataset preparator.
        Args:
            output_dir: Directory to save processed dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(exist_ok=True)
            (dir_path / "images").mkdir(exist_ok=True)
            (dir_path / "masks").mkdir(exist_ok=True)
    
    def process_roofline_dataset(self, matlab_path):
        """
        Process Roofline-Extraction dataset from MATLAB file.
        Args:
            matlab_path: Path to imdb.mat file
        """
        print(f"Processing Roofline-Extraction dataset from {matlab_path}")
        
        try:
            # Load MATLAB file
            data = sio.loadmat(matlab_path)
            imdb = data['imdb']
            
            # Extract data
            images = imdb['images'][0, 0]  # RGB images
            depths = imdb['depths'][0, 0]  # nDSMs
            elements = imdb['elements'][0, 0]  # Roof elements
            
            num_samples = len(images)
            print(f"Found {num_samples} samples")
            
            # Split indices
            indices = np.random.permutation(num_samples)
            train_size = int(num_samples * DATASET_CONFIG['train_split'])
            val_size = int(num_samples * DATASET_CONFIG['val_split'])
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size+val_size]
            test_indices = indices[train_size+val_size:]
            
            # Process each split
            splits = [
                (train_indices, self.train_dir, 'train'),
                (val_indices, self.val_dir, 'validation'),
                (test_indices, self.test_dir, 'test')
            ]
            
            for indices, output_dir, split_name in splits:
                print(f"\nProcessing {split_name} split...")
                for idx in tqdm(indices):
                    # Save image
                    image = images[idx]
                    image_path = output_dir / "images" / f"roofline_{idx:05d}.png"
                    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    
                    # Create feature masks
                    element_mask = elements[idx]
                    h, w = element_mask.shape[:2]
                    
                    # Initialize separate masks for each feature
                    outline_mask = np.zeros((h, w), dtype=np.uint8)
                    ridge_mask = np.zeros((h, w), dtype=np.uint8)
                    hip_mask = np.zeros((h, w), dtype=np.uint8)
                    valley_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # Extract features based on color coding
                    outline_mask[element_mask[..., 0] > 128] = 255  # Red channel for eaves
                    ridge_mask[element_mask[..., 1] > 128] = 255    # Green channel for ridges
                    hip_mask[element_mask[..., 2] > 128] = 255      # Blue channel for hips
                    
                    # Save masks
                    mask_dir = output_dir / "masks" / f"roofline_{idx:05d}"
                    mask_dir.mkdir(exist_ok=True)
                    
                    cv2.imwrite(str(mask_dir / "outline.png"), outline_mask)
                    cv2.imwrite(str(mask_dir / "ridge.png"), ridge_mask)
                    cv2.imwrite(str(mask_dir / "hip.png"), hip_mask)
                    cv2.imwrite(str(mask_dir / "valley.png"), valley_mask)
            
            print("Roofline-Extraction dataset processing complete")
            
        except Exception as e:
            print(f"Error processing Roofline-Extraction dataset: {str(e)}")
            raise
    
    def process_rid_dataset(self, rid_dir):
        """
        Process RID dataset.
        Args:
            rid_dir: Directory containing RID dataset
        """
        print(f"Processing RID dataset from {rid_dir}")
        
        try:
            rid_dir = Path(rid_dir)
            image_dir = rid_dir / "images"
            label_dir = rid_dir / "labels"
            
            if not image_dir.exists() or not label_dir.exists():
                raise ValueError(f"Invalid RID dataset directory structure in {rid_dir}")
            
            # Get all image files
            image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            num_samples = len(image_files)
            print(f"Found {num_samples} samples")
            
            # Split indices
            indices = np.random.permutation(num_samples)
            train_size = int(num_samples * DATASET_CONFIG['train_split'])
            val_size = int(num_samples * DATASET_CONFIG['val_split'])
            
            train_files = [image_files[i] for i in indices[:train_size]]
            val_files = [image_files[i] for i in indices[train_size:train_size+val_size]]
            test_files = [image_files[i] for i in indices[train_size+val_size:]]
            
            # Process each split
            splits = [
                (train_files, self.train_dir, 'train'),
                (val_files, self.val_dir, 'validation'),
                (test_files, self.test_dir, 'test')
            ]
            
            for files, output_dir, split_name in splits:
                print(f"\nProcessing {split_name} split...")
                for image_file in tqdm(files):
                    # Load and save image
                    image = cv2.imread(str(image_file))
                    image_path = output_dir / "images" / f"rid_{image_file.stem}.png"
                    cv2.imwrite(str(image_path), image)
                    
                    # Load and process label
                    label_file = label_dir / f"{image_file.stem}_label.png"
                    if label_file.exists():
                        label = cv2.imread(str(label_file))
                        
                        # Create feature masks based on RID labels
                        mask_dir = output_dir / "masks" / f"rid_{image_file.stem}"
                        mask_dir.mkdir(exist_ok=True)
                        
                        # Convert RID labels to our format
                        outline_mask = np.zeros(label.shape[:2], dtype=np.uint8)
                        ridge_mask = np.zeros(label.shape[:2], dtype=np.uint8)
                        hip_mask = np.zeros(label.shape[:2], dtype=np.uint8)
                        valley_mask = np.zeros(label.shape[:2], dtype=np.uint8)
                        
                        # Map RID labels to our features
                        # Note: Adjust these mappings based on RID's actual label format
                        outline_mask[label[..., 0] > 128] = 255
                        ridge_mask[label[..., 1] > 128] = 255
                        hip_mask[label[..., 2] > 128] = 255
                        
                        # Save masks
                        cv2.imwrite(str(mask_dir / "outline.png"), outline_mask)
                        cv2.imwrite(str(mask_dir / "ridge.png"), ridge_mask)
                        cv2.imwrite(str(mask_dir / "hip.png"), hip_mask)
                        cv2.imwrite(str(mask_dir / "valley.png"), valley_mask)
            
            print("RID dataset processing complete")
            
        except Exception as e:
            print(f"Error processing RID dataset: {str(e)}")
            raise

def main():
    """Main function to prepare datasets."""
    import argparse
    parser = argparse.ArgumentParser(description="Prepare roof detection datasets")
    parser.add_argument("--roofline", help="Path to Roofline-Extraction imdb.mat file")
    parser.add_argument("--rid", help="Path to RID dataset directory")
    parser.add_argument("--output", default="data/processed",
                       help="Output directory for processed dataset")
    args = parser.parse_args()
    
    preparator = DatasetPreparator(args.output)
    
    if args.roofline:
        preparator.process_roofline_dataset(args.roofline)
    
    if args.rid:
        preparator.process_rid_dataset(args.rid)
    
    if not args.roofline and not args.rid:
        print("Error: Please provide at least one dataset path (--roofline or --rid)")

if __name__ == "__main__":
    main()
