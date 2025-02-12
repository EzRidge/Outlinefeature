import logging
import torch
from .rid_dataset import RIDDataset
from .roofline_dataset import RooflineDataset

def create_dataloaders(dataset_path, dataset_type, batch_size=4, num_workers=4, max_samples=None):
    """
    Create training and validation dataloaders for a specific dataset.
    
    Args:
        dataset_path (str): Path to dataset
        dataset_type (str): Type of dataset ('rid', 'roofline', or 'airs')
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
        max_samples (int): Maximum number of samples to load (for testing)
        
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    logging.info(f"Creating dataloaders for {dataset_type} dataset")
    logging.info(f"Dataset path: {dataset_path}")
    if max_samples:
        logging.info(f"Using max_samples={max_samples} for testing")
    
    # Select dataset class based on type
    dataset_classes = {
        'rid': RIDDataset,
        'roofline': RooflineDataset,
        # 'airs': AIRSDataset,  # To be added
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: {list(dataset_classes.keys())}")
    
    dataset_class = dataset_classes[dataset_type]
    
    # Create datasets
    train_dataset = dataset_class(dataset_path, split='train', max_samples=max_samples)
    val_dataset = dataset_class(dataset_path, split='val', max_samples=max_samples)
    
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