"""
Script to prepare unified training data from multiple roof datasets:
- RID (Roof Information Dataset)
- Roofline-Extraction Dataset
- AIRS Dataset
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class UnifiedDatasetPreparation:
    def __init__(self, output_path, split_ratio=0.8):
        """
        Initialize unified dataset preparation.
        
        Args:
            output_path (str): Path to save processed dataset
            split_ratio (float): Train/val split ratio
        """
        self.output_path = Path(output_path)
        self.split_ratio = split_ratio
        
        # Unified class mapping for all datasets
        self.unified_class_mapping = {
            # Core classes (present in all datasets)
            'roof': 1,
            'ridge': 2,
            'valley': 3,
            'eave': 4,
            
            # Additional features
            'dormer': 5,
            'chimney': 6,
            'window': 7,
            'pv_module': 8,
            'shadow': 9,
            'tree': 10,
            'unknown': 11
        }
        
        # Create output directories
        self.train_img_dir = self.output_path / 'train' / 'images'
        self.train_mask_dir = self.output_path / 'train' / 'masks'
        self.val_img_dir = self.output_path / 'val' / 'images'
        self.val_mask_dir = self.output_path / 'val' / 'masks'
        
        for dir_path in [self.train_img_dir, self.train_mask_dir, 
                        self.val_img_dir, self.val_mask_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_mask_from_rid_annotations(self, image_size, annotations):
        """Create segmentation mask from RID annotations."""
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # RID to unified class mapping
        rid_mapping = {
            'roof': 'roof',
            'ridge': 'ridge',
            'eave': 'eave'
        }
        
        for annotation in annotations:
            class_name = annotation['class']
            if class_name in rid_mapping:
                unified_class = rid_mapping[class_name]
                points = np.array(annotation['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], self.unified_class_mapping[unified_class])
        
        return mask

    def create_mask_from_roofline_annotations(self, image_size, annotations):
        """Create segmentation mask from Roofline-Extraction annotations."""
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # Roofline to unified class mapping
        roofline_mapping = {
            'outline': 'roof',
            'ridge_line': 'ridge',
            'valley_line': 'valley',
            'eave_line': 'eave'
        }
        
        for annotation in annotations:
            class_name = annotation['type']
            if class_name in roofline_mapping:
                unified_class = roofline_mapping[class_name]
                points = np.array(annotation['points'], dtype=np.int32)
                cv2.polylines(mask, [points], True, self.unified_class_mapping[unified_class], 2)
        
        return mask

    def create_mask_from_airs_annotations(self, image_size, annotations):
        """Create segmentation mask from AIRS annotations."""
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # AIRS to unified class mapping
        airs_mapping = {
            'building': 'roof',
            'ridge': 'ridge',
            'valley': 'valley',
            'eave': 'eave'
        }
        
        for annotation in annotations:
            class_name = annotation['category']
            if class_name in airs_mapping:
                unified_class = airs_mapping[class_name]
                points = np.array(annotation['segmentation'], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], self.unified_class_mapping[unified_class])
        
        return mask

    def process_dataset(self, dataset_path, dataset_type):
        """
        Process a specific dataset.
        
        Args:
            dataset_path (str): Path to dataset
            dataset_type (str): Type of dataset ('rid', 'roofline', or 'airs')
        """
        dataset_path = Path(dataset_path)
        print(f'\nProcessing {dataset_type.upper()} dataset from {dataset_path}')
        
        # Get image paths
        image_paths = list(dataset_path.glob('**/*.jpg'))
        image_paths.extend(dataset_path.glob('**/*.png'))
        np.random.shuffle(image_paths)
        
        # Split into train/val
        split_idx = int(len(image_paths) * self.split_ratio)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        
        def process_subset(image_paths, img_dir, mask_dir):
            for img_path in tqdm(image_paths, desc=f'Processing {dataset_type}'):
                # Load and resize image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                image = cv2.resize(image, (1024, 1024))
                
                # Load and process annotations based on dataset type
                anno_path = img_path.with_suffix('.json')
                if not anno_path.exists():
                    continue
                    
                with open(anno_path, 'r') as f:
                    annotations = json.load(f)
                
                # Create mask based on dataset type
                if dataset_type == 'rid':
                    mask = self.create_mask_from_rid_annotations((1024, 1024), annotations)
                elif dataset_type == 'roofline':
                    mask = self.create_mask_from_roofline_annotations((1024, 1024), annotations)
                else:  # airs
                    mask = self.create_mask_from_airs_annotations((1024, 1024), annotations)
                
                # Save files with dataset prefix to avoid name collisions
                base_name = f"{dataset_type}_{img_path.stem}"
                cv2.imwrite(str(img_dir / f'{base_name}.jpg'), image)
                cv2.imwrite(str(mask_dir / f'{base_name}_mask.png'), mask)
        
        # Process train and val sets
        process_subset(train_paths, self.train_img_dir, self.train_mask_dir)
        process_subset(val_paths, self.val_img_dir, self.val_mask_dir)
        
        print(f'Processed {len(train_paths)} training and {len(val_paths)} validation images')

    def save_metadata(self):
        """Save dataset metadata and class mapping."""
        metadata = {
            'class_mapping': self.unified_class_mapping,
            'image_size': 1024,
            'num_classes': len(self.unified_class_mapping)
        }
        
        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def prepare_all_datasets(rid_path, roofline_path, airs_path, output_path, split_ratio=0.8):
    """
    Prepare all datasets in unified format.
    
    Args:
        rid_path (str): Path to RID dataset
        roofline_path (str): Path to Roofline-Extraction dataset
        airs_path (str): Path to AIRS dataset
        output_path (str): Path to save processed dataset
        split_ratio (float): Train/val split ratio
    """
    processor = UnifiedDatasetPreparation(output_path, split_ratio)
    
    # Process each dataset
    if rid_path and Path(rid_path).exists():
        processor.process_dataset(rid_path, 'rid')
    
    if roofline_path and Path(roofline_path).exists():
        processor.process_dataset(roofline_path, 'roofline')
    
    if airs_path and Path(airs_path).exists():
        processor.process_dataset(airs_path, 'airs')
    
    # Save metadata
    processor.save_metadata()
    print('\nDataset preparation complete!')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare unified roof dataset for training')
    parser.add_argument('--rid-path', help='Path to RID dataset')
    parser.add_argument('--roofline-path', help='Path to Roofline-Extraction dataset')
    parser.add_argument('--airs-path', help='Path to AIRS dataset')
    parser.add_argument('--output-path', required=True, help='Path to save processed dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Train/val split ratio')
    
    args = parser.parse_args()
    
    prepare_all_datasets(
        args.rid_path,
        args.roofline_path,
        args.airs_path,
        args.output_path,
        args.split
    )