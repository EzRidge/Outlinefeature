"""
Configuration settings for the hybrid roof feature detection model.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data/output"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    # Basic model settings
    'model_name': 'hybrid_roof_detector',
    'backbone': 'resnet50',
    'pretrained': True,
    
    # Input settings
    'input_size': (1024, 1024),  # Based on paper recommendations
    'input_channels': 3,  # RGB input
    
    # Output settings
    'num_classes': 4,  # outline, ridge, hip, valley
    'feature_channels': 64,  # Base number of feature channels
    
    # Training settings
    'batch_size': 8,  # Optimized for g5.2xlarge
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    
    # Data augmentation
    'augmentation': {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotate': True,
        'rotate_limit': 45,
        'brightness_contrast': True
    }
}

# Training configuration
TRAIN_CONFIG = {
    'device': 'cuda',  # or 'cpu'
    'num_workers': 4,
    'pin_memory': True,
    'save_frequency': 5,  # Save model every N epochs
    
    # Early stopping
    'early_stopping': {
        'patience': 10,
        'min_delta': 1e-4
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'log_dir': PROJECT_ROOT / "logs",
    'tensorboard': True,
    'log_frequency': 100,  # Log every N batches
    'save_images': True,
    'num_images_to_save': 5
}

# Create logging directory
LOGGING_CONFIG['log_dir'].mkdir(parents=True, exist_ok=True)
