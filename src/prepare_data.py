"""
Script to prepare training data from the RID (Roof Information Dataset).
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def create_mask_from_annotations(image_size, annotations, class_mapping):
    """
    Create segmentation mask from RID annotations.
    
    Args:
        image_size (tuple): Size of the image (height, width)
        annotations (dict): Dictionary containing annotations
        class_mapping (dict): Mapping from annotation classes to mask values
        
    Returns:
        np.ndarray: Segmentation mask
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # Draw each annotation onto the mask
    for annotation in annotations:
        class_name = annotation['class']
        if class_name in class_mapping:
            points = np.array(annotation['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], class_mapping[class_name])
    
    return mask

def prepare_dataset(rid_path, output_path, split_ratio=0.8):
    """
    Prepare training dataset from RID.
    
    Args:
        rid_path (str): Path to RID dataset
        output_path (str): Path to save processed dataset
        split_ratio (float): Train/val split ratio
    """
    rid_path = Path(rid_path)
    output_path = Path(output_path)
    
    # Create output directories
    train_img_dir = output_path / 'train' / 'images'
    train_mask_dir = output_path / 'train' / 'masks'
    val_img_dir = output_path / 'val' / 'images'
    val_mask_dir = output_path / 'val' / 'masks'
    
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Class mapping for mask values
    class_mapping = {
        'roof': 1,
        'ridge': 2,
        'eave': 3
    }
    
    # Process each image
    image_paths = list(rid_path.glob('*.jpg'))
    np.random.shuffle(image_paths)
    
    # Split into train/val
    split_idx = int(len(image_paths) * split_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    def process_subset(image_paths, img_dir, mask_dir):
        for img_path in tqdm(image_paths):
            # Load and resize image
            image = cv2.imread(str(img_path))
            image = cv2.resize(image, (1024, 1024))
            
            # Load annotations
            anno_path = img_path.with_suffix('.json')
            if not anno_path.exists():
                continue
                
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            
            # Create mask
            mask = create_mask_from_annotations(
                (1024, 1024),
                annotations,
                class_mapping
            )
            
            # Save files
            base_name = img_path.stem
            cv2.imwrite(str(img_dir / f'{base_name}.jpg'), image)
            cv2.imwrite(str(mask_dir / f'{base_name}_mask.png'), mask)
    
    # Process train and val sets
    print('Processing training set...')
    process_subset(train_paths, train_img_dir, train_mask_dir)
    
    print('Processing validation set...')
    process_subset(val_paths, val_img_dir, val_mask_dir)
    
    # Save class mapping
    with open(output_path / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f'Dataset prepared with {len(train_paths)} training and {len(val_paths)} validation images')

def verify_dataset(dataset_path):
    """
    Verify the prepared dataset.
    
    Args:
        dataset_path (str): Path to processed dataset
    """
    dataset_path = Path(dataset_path)
    
    def check_subset(subset):
        img_dir = dataset_path / subset / 'images'
        mask_dir = dataset_path / subset / 'masks'
        
        n_images = len(list(img_dir.glob('*.jpg')))
        n_masks = len(list(mask_dir.glob('*.png')))
        
        print(f'{subset} set:')
        print(f'  Images: {n_images}')
        print(f'  Masks: {n_masks}')
        
        if n_images != n_masks:
            print('  Warning: Number of images and masks does not match!')
        
        # Check image sizes
        for img_path in img_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            mask_path = mask_dir / f'{img_path.stem}_mask.png'
            
            if not mask_path.exists():
                print(f'  Warning: Missing mask for {img_path.name}')
                continue
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img.shape[:2] != mask.shape:
                print(f'  Warning: Size mismatch for {img_path.name}')
                print(f'    Image size: {img.shape[:2]}')
                print(f'    Mask size: {mask.shape}')
    
    check_subset('train')
    check_subset('val')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare RID dataset for training')
    parser.add_argument('rid_path', help='Path to RID dataset')
    parser.add_argument('output_path', help='Path to save processed dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--verify', action='store_true', help='Verify dataset after preparation')
    
    args = parser.parse_args()
    
    prepare_dataset(args.rid_path, args.output_path, args.split)
    
    if args.verify:
        verify_dataset(args.output_path)
