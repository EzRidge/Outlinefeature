import os
import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader

# Import our updated components
from models import create_model
from data import UnifiedRoofDataset, create_dataloaders
from result_manager import ResultManager

def parse_args():
    """Parse SageMaker training arguments"""
    parser = argparse.ArgumentParser()
    
    # Data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--val-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=12)
    
    # Parse args
    args, _ = parser.parse_known_args()
    return args

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        predictions = model(images)
        loss, losses_dict = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logging.info(f'Train Batch [{batch_idx + 1}/{num_batches}] - Loss: {loss.item():.4f}')
            for k, v in losses_dict.items():
                logging.info(f'  {k}: {v.item():.4f}')
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, targets in val_loader:
            # Move data to device
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            
            # Update total loss
            total_loss += loss.item()
    
    return total_loss / num_batches

def train(args):
    """Training function for SageMaker"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # Initialize result manager
        result_manager = ResultManager(os.path.join(args.model_dir, 'results'))
        
        # Create dataloaders
        logging.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            args.train_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Initialize model and training components
        logging.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, criterion = create_model(num_classes=args.num_classes)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        logging.info(f"Starting training on {device}")
        logging.info(f"Training samples: {len(train_loader.dataset)}")
        logging.info(f"Validation samples: {len(val_loader.dataset)}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Number of epochs: {args.epochs}")
        
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            logging.info(f"\nEpoch [{epoch + 1}/{args.epochs}]")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            logging.info(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            logging.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, os.path.join(args.model_dir, 'best_model.pth'))
            
            # Save metrics
            result_manager.save_metrics({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        # Save final model
        final_checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss
        }
        torch.save(final_checkpoint, os.path.join(args.model_dir, 'final_model.pth'))
        
        logging.info(f"\nTraining completed!")
        logging.info(f"Best validation loss: {best_val_loss:.4f}")
        logging.info(f"Model saved to: {args.model_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    args = parse_args()
    train(args)