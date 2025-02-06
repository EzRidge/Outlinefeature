import os
import logging
from typing import List, Tuple

def get_dataset_splits(base_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Read train/val/test splits from files"""
    split_path = os.path.join(base_path, "filenames_train_val_test_split")
    
    def read_split_file(filename: str) -> List[str]:
        filepath = os.path.join(split_path, filename)
        if not os.path.exists(filepath):
            logging.error(f"Split file not found: {filepath}")
            return []
        with open(filepath, 'r') as f:
            # Remove any image extension (.png or .tif) and strip whitespace
            return [line.strip().replace('.png', '').replace('.tif', '') 
                   for line in f.readlines()]
    
    # Using split 1 for consistency
    train_samples = read_split_file("train_filenames_1_rev.txt")
    val_samples = read_split_file("val_filenames_1_rev.txt")
    test_samples = read_split_file("test_filenames_1_rev.txt")
    
    logging.info(f"Found {len(train_samples)} training samples")
    logging.info(f"Found {len(val_samples)} validation samples")
    logging.info(f"Found {len(test_samples)} test samples")
    
    return train_samples, val_samples, test_samples

def get_training_subset(train_samples: List[str], 
                       val_samples: List[str], 
                       train_size: int = 16,  # Reduced for quick testing
                       val_size: int = 4) -> Tuple[List[str], List[str]]:  # Reduced for quick testing
    """Get subset of samples for training"""
    train_subset = train_samples[:train_size]
    val_subset = val_samples[:val_size]
    
    logging.info(f"Using {len(train_subset)} training samples")
    logging.info(f"Using {len(val_subset)} validation samples")
    
    return train_subset, val_subset

class TrainingConfig:
    def __init__(self):
        # Data paths
        self.rid_path = os.path.join("Reference Materials", "data", "RID", "m1655470", "RID_dataset")
        self.roofline_path = os.path.join("Reference Materials", "data", "Roofline-Extraction")
        
        # Training parameters
        self.num_epochs = 5  # Reduced for quick testing
        self.batch_size = 4  # Reduced for quick testing
        self.learning_rate = 0.001
        self.val_interval = 1  # Validate and visualize every epoch
        
        # Model parameters
        self.num_segment_classes = 9  # Fixed to match our class mapping
        
        # Results
        self.results_dir = "results"

def verify_paths(config: TrainingConfig) -> bool:
    """Verify all required paths exist"""
    paths_to_check = [
        config.rid_path,
        config.roofline_path,
        os.path.join(config.rid_path, "filenames_train_val_test_split"),
        os.path.join(config.rid_path, "images_roof_centered_geotiff"),
        os.path.join(config.rid_path, "masks_segments_reviewed"),
        os.path.join(config.rid_path, "masks_superstructures_reviewed")
    ]
    
    all_exist = True
    for path in paths_to_check:
        if not os.path.exists(path):
            logging.error(f"Required path does not exist: {path}")
            all_exist = False
    
    if all_exist:
        logging.info("All required paths exist")
        
        # Also verify split files exist
        split_files = [
            os.path.join(config.rid_path, "filenames_train_val_test_split", "train_filenames_1_rev.txt"),
            os.path.join(config.rid_path, "filenames_train_val_test_split", "val_filenames_1_rev.txt"),
            os.path.join(config.rid_path, "filenames_train_val_test_split", "test_filenames_1_rev.txt")
        ]
        for file in split_files:
            if not os.path.exists(file):
                logging.error(f"Split file not found: {file}")
                all_exist = False
    
    return all_exist

def setup_training(train_size: int = 16, val_size: int = 4) -> Tuple[TrainingConfig, List[str], List[str]]:  # Updated defaults
    """Setup training configuration and get data splits"""
    config = TrainingConfig()
    
    # Verify paths
    if not verify_paths(config):
        raise RuntimeError("Required paths not found. Please check the paths in TrainingConfig.")
    
    # Get dataset splits
    train_samples, val_samples, _ = get_dataset_splits(config.rid_path)
    
    # Get training subset
    train_subset, val_subset = get_training_subset(
        train_samples,
        val_samples,
        train_size=train_size,
        val_size=val_size
    )
    
    return config, train_subset, val_subset

if __name__ == "__main__":
    # Test the setup
    logging.basicConfig(level=logging.INFO)
    try:
        config, train_subset, val_subset = setup_training()
        print("\nSetup successful!")
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
    except Exception as e:
        print(f"\nSetup failed: {str(e)}")
