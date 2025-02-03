"""
Main script for hybrid roof feature detection system.
Provides command-line interface for all operations.
"""

import argparse
import sys
from pathlib import Path
from .prepare_data import DatasetPreparator
from .train import main as train_main
from .quick_test import process_single_image, process_directory
from .models import create_model, load_pretrained
from .config import MODEL_CONFIG, TRAIN_CONFIG

def setup_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Hybrid Roof Feature Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare datasets')
    prepare_parser.add_argument('--roofline', help='Path to Roofline-Extraction imdb.mat file')
    prepare_parser.add_argument('--rid', help='Path to RID dataset directory')
    prepare_parser.add_argument('--output', default='data/processed',
                              help='Output directory for processed dataset')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', default='data/processed',
                            help='Directory containing processed dataset')
    train_parser.add_argument('--weights', help='Path to pretrained weights')
    train_parser.add_argument('--resume', help='Path to checkpoint to resume from')
    train_parser.add_argument('--config', help='Path to custom configuration file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('input_path', help='Path to input image or directory')
    test_parser.add_argument('--weights', required=True,
                           help='Path to model weights')
    test_parser.add_argument('--output', help='Path to output directory')
    test_parser.add_argument('--batch', action='store_true',
                           help='Process entire directory')
    
    return parser

def prepare_command(args):
    """Handle prepare command."""
    if not args.roofline and not args.rid:
        print("Error: Please provide at least one dataset path (--roofline or --rid)")
        return 1
    
    try:
        preparator = DatasetPreparator(args.output)
        
        if args.roofline:
            preparator.process_roofline_dataset(args.roofline)
        
        if args.rid:
            preparator.process_rid_dataset(args.rid)
        
        return 0
        
    except Exception as e:
        print(f"Error preparing datasets: {str(e)}")
        return 1

def train_command(args):
    """Handle train command."""
    try:
        # Update configuration if custom config provided
        if args.config:
            import yaml
            with open(args.config) as f:
                custom_config = yaml.safe_load(f)
                MODEL_CONFIG.update(custom_config.get('model_config', {}))
                TRAIN_CONFIG.update(custom_config.get('train_config', {}))
        
        # Run training
        return train_main(args)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 1

def test_command(args):
    """Handle test command."""
    try:
        # Create and initialize model
        print("Initializing model...")
        model = create_model()
        
        if not Path(args.weights).exists():
            print(f"Error: Weights file not found at {args.weights}")
            return 1
        
        print(f"Loading weights from {args.weights}")
        model = load_pretrained(model, args.weights)
        model.eval()
        
        # Create output directory
        output_dir = Path(args.output) if args.output else Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process input
        input_path = Path(args.input_path)
        if args.batch or input_path.is_dir():
            process_directory(input_path, model, output_dir, args.weights)
        else:
            process_single_image(input_path, model, output_dir, args.weights)
        
        return 0
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return 1

def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute appropriate command
    command_handlers = {
        'prepare': prepare_command,
        'train': train_command,
        'test': test_command
    }
    
    return command_handlers[args.command](args)

if __name__ == "__main__":
    sys.exit(main())
