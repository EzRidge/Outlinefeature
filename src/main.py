"""
Main script for roof feature detection and measurement.
"""

import os
import sys
import logging
from pathlib import Path

from .prepare_data import prepare_dataset, verify_dataset
from .train import train_model
from .predict import main as predict_main

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('roof_detection.log')
        ]
    )
    return logging.getLogger(__name__)

def prepare_command(args):
    """Handle dataset preparation command."""
    logger = setup_logging()
    logger.info('Preparing dataset...')
    
    prepare_dataset(args.rid_path, args.output_path, args.split)
    
    if args.verify:
        logger.info('Verifying dataset...')
        verify_dataset(args.output_path)
    
    logger.info('Dataset preparation completed')

def train_command(args):
    """Handle model training command."""
    logger = setup_logging()
    logger.info('Starting model training...')
    
    train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    logger.info('Training completed')

def predict_command(args):
    """Handle prediction command."""
    logger = setup_logging()
    logger.info('Running inference...')
    
    predict_main(
        args.model_path,
        args.input_path,
        args.output_dir,
        args.device
    )
    
    logger.info('Inference completed')

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Roof feature detection and measurement')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dataset preparation command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare training dataset')
    prepare_parser.add_argument('rid_path', help='Path to RID dataset')
    prepare_parser.add_argument('output_path', help='Path to save processed dataset')
    prepare_parser.add_argument('--split', type=float, default=0.8, help='Train/val split ratio')
    prepare_parser.add_argument('--verify', action='store_true', help='Verify dataset after preparation')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('image_dir', help='Directory containing training images')
    train_parser.add_argument('mask_dir', help='Directory containing mask annotations')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--device', default='cuda', help='Device to train on')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Run inference')
    predict_parser.add_argument('model_path', help='Path to trained model checkpoint')
    predict_parser.add_argument('input_path', help='Path to input image or directory')
    predict_parser.add_argument('output_dir', help='Directory to save results')
    predict_parser.add_argument('--device', default='cuda', help='Device to run inference on')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
