"""
Configuration settings for the hybrid roof feature detection model.
Combines RID's segmentation approach with Roofline-Extraction's feature detection.
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
    'input_size': (1024, 1024),  # Based on Roofline-Extraction paper
    'input_channels': 3,  # RGB input
    
    # Output settings
    'num_classes': 4,  # outline, ridge, hip, valley
    'feature_channels': 64,  # Base number of feature channels
    
    # Training settings
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    
    # Data augmentation
    'augmentation': {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotate': True,
        'rotate_limit': 45,
        'brightness_contrast': True,
        'gaussian_noise': True
    },
    
    # Feature detection thresholds
    'detection': {
        'outline_threshold': 0.5,
        'ridge_threshold': 0.3,
        'hip_threshold': 0.3,
        'valley_threshold': 0.3,
        'min_area_ratio': 0.01,
        'max_angle_deviation': 10  # degrees
    }
}

# Dataset configuration
DATASET_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    
    # Class weights for loss function
    'class_weights': {
        'outline': 1.0,
        'ridge': 1.2,
        'hip': 1.2,
        'valley': 1.2
    },
    
    # Color mapping for visualization
    'color_map': {
        'outline': (0, 255, 0),    # Green
        'ridge': (255, 0, 0),      # Red
        'hip': (0, 0, 255),        # Blue
        'valley': (255, 255, 0)    # Yellow
    }
}

# Training configuration
TRAIN_CONFIG = {
    'device': 'cuda',  # or 'cpu'
    'num_workers': 4,
    'pin_memory': True,
    'save_frequency': 5,  # Save model every N epochs
    
    # Loss function weights
    'loss_weights': {
        'segmentation': 1.0,
        'boundary': 0.5,
        'angle': 0.3
    },
    
    # Early stopping
    'early_stopping': {
        'patience': 10,
        'min_delta': 1e-4
    }
}

# Inference configuration
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'nms_threshold': 0.3,
    'min_line_length': 20,
    'max_line_gap': 10,
    
    # Post-processing
    'post_processing': {
        'smooth_outline': True,
        'merge_lines': True,
        'angle_snap': True,
        'angle_threshold': 5  # degrees
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