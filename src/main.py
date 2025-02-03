"""
Main script for training and evaluating the roof feature detection model.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .models import create_model
from .data import RoofDataset
from .train import train_model
from .utils import visualize_predictions, save_predictions
from .config import MODEL_CONFIG, TRAIN_CONFIG, DATA_DIR, MODELS_DIR

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Roof Feature Detection')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                      help='Run mode: train or eval')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR),
                      help='Path to data directory')
    parser.add_argument('--model_path', type=str, default=str(MODELS_DIR / 'best_model.pth'),
                      help='Path to model weights (for eval mode)')
    parser.add_argument('--batch_size', type=int, default=MODEL_CONFIG['batch_size'],
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['epochs'],
                      help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=MODEL_CONFIG['learning_rate'],
                      help='Learning rate')
    return parser.parse_args()

def train(args):
    """Training pipeline."""
    logging.info('Starting training pipeline')
    
    # Create model
    model = create_model(MODEL_CONFIG)
    logging.info(f'Created model: {model.__class__.__name__}')
    
    # Setup data loaders
    train_dataset = RoofDataset(split='train')
    val_dataset = RoofDataset(split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    # Training
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    logging.info('Training completed')

def evaluate(args):
    """Evaluation pipeline."""
    logging.info('Starting evaluation pipeline')
    
    # Load model
    model = create_model(MODEL_CONFIG)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(TRAIN_CONFIG['device'])
    model.eval()
    logging.info(f'Loaded model from {args.model_path}')
    
    # Setup data loader
    test_dataset = RoofDataset(split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    # Evaluate
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(TRAIN_CONFIG['device'])
            predictions = model(images)
            
            # Save predictions
            for i in range(images.size(0)):
                save_predictions(
                    images[i],
                    {k: v[i:i+1] for k, v in predictions.items()},
                    f'test_sample_{batch_idx}_{i}'
                )
    
    logging.info('Evaluation completed')

def main():
    """Main function."""
    args = parse_args()
    setup_logging()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main()
