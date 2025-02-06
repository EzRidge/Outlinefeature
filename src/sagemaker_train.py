import os
import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader

# Import our existing components
from src.models import HybridRoofModel, RoofLoss
from src.data import RoofDataset
from src.prepare_training import setup_training
from src.train import train_model
from src.result_manager import ResultManager

def parse_args():
    """Parse SageMaker training arguments"""
    parser = argparse.ArgumentParser()
    
    # Data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--val-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Parse args
    args, _ = parser.parse_known_args()
    return args

def train(args):
    """Training function for SageMaker"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # Setup training configuration and data splits
        logging.info("Setting up training configuration...")
        config, train_subset, val_subset = setup_training(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        # Initialize result manager (save to SageMaker paths)
        result_manager = ResultManager(os.path.join(args.model_dir, 'results'))
        
        # Create datasets
        logging.info("Creating datasets...")
        train_dataset = RoofDataset(train_subset)
        val_dataset = RoofDataset(val_subset)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Initialize model and training components
        logging.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridRoofModel(num_segment_classes=config.num_segment_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = RoofLoss()
        
        # Train model
        logging.info(f"Starting training on {device}")
        logging.info(f"Training samples: {len(train_subset)}")
        logging.info(f"Validation samples: {len(val_subset)}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Number of epochs: {args.epochs}")
        logging.info(f"Results will be saved in: {result_manager.run_dir}")
        
        model, best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            result_manager=result_manager
        )
        
        # Save final model artifacts
        model_path = os.path.join(args.model_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'best_val_loss': best_val_loss
        }, model_path)
        
        logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        logging.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    args = parse_args()
    train(args)