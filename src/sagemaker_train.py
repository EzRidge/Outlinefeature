import os
import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader

# Import our updated components
from models import create_model
from data import create_dataloaders
from result_manager import ResultManager

def parse_args():
    """Parse SageMaker training arguments"""
    parser = argparse.ArgumentParser()
    
    # Data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--rid-dir', type=str, default=os.environ.get('SM_CHANNEL_RID', None))
    parser.add_argument('--roofline-dir', type=str, default=os.environ.get('SM_CHANNEL_ROOFLINE', None))
    parser.add_argument('--airs-dir', type=str, default=os.environ.get('SM_CHANNEL_AIRS', None))
    
    # Training hyperparameters
    parser.add_argument('--epochs-per-dataset', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=12)
    
    # Parse args
    args, _ = parser.parse_known_args()
    return args

def list_directory_contents(path, indent=""):
    """Helper function to recursively list directory contents"""
    if not os.path.exists(path):
        return f"{path} does not exist"
    
    result = []
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result.append(f"{indent}{item}/")
                result.extend(list_directory_contents(item_path, indent + "  "))
            else:
                result.append(f"{indent}{item}")
    except Exception as e:
        result.append(f"Error reading {path}: {str(e)}")
    
    return result

def train_epoch(model, train_loader, optimizer, device, dataset_type):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        # Only move tensor values to device
        device_targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }
        
        # Forward pass
        predictions = model(images)
        loss, losses_dict = model.criterion(predictions, device_targets)
        
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

def validate(model, val_loader, device, dataset_type):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, targets in val_loader:
            # Move data to device
            images = images.to(device)
            # Only move tensor values to device
            device_targets = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in targets.items()
            }
            
            # Forward pass
            predictions = model(images)
            loss, _ = model.criterion(predictions, device_targets)
            
            # Update total loss
            total_loss += loss.item()
    
    return total_loss / num_batches

def train_on_dataset(model, dataset_path, dataset_type, args, device, start_epoch=0):
    """Train model on a specific dataset"""
    logging.info(f"\nStarting training on {dataset_type} dataset")
    logging.info(f"Dataset path: {dataset_path}")
    logging.info("Directory contents:")
    contents = list_directory_contents(dataset_path)
    for line in contents:
        logging.info(line)
    
    # Create dataloaders for this dataset
    train_loader, val_loader = create_dataloaders(
        dataset_path,
        dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Adjust learning rate based on dataset phase
    lr_multiplier = {
        'rid': 1.0,      # Base learning rate for initial training
        'roofline': 0.1, # Lower learning rate for fine-tuning on lines
        'airs': 0.05     # Even lower for final outline tuning
    }
    
    # Initialize optimizer with adjusted learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate * lr_multiplier[dataset_type]
    )
    
    # Training loop for this dataset
    best_val_loss = float('inf')
    epochs = args.epochs_per_dataset
    
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch
        logging.info(f"\nEpoch [{epoch + 1}/{epochs}] ({dataset_type})")
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, device, dataset_type)
        val_loss = validate(model, val_loader, device, dataset_type)
        
        logging.info(f"Training Loss: {train_loss:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model for this dataset
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': current_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'dataset_type': dataset_type
            }
            checkpoint_path = os.path.join(args.model_dir, f'best_model_{dataset_type}.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved best model for {dataset_type}")
    
    return current_epoch + 1

def train(args):
    """Sequential training on multiple datasets"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # Log all input directories
        logging.info("Input directories:")
        logging.info(f"RID dir: {args.rid_dir}")
        logging.info(f"Roofline dir: {args.roofline_dir}")
        logging.info(f"AIRS dir: {args.airs_dir}")
        
        # Initialize result manager
        result_manager = ResultManager(os.path.join(args.model_dir, 'results'))
        
        # Initialize model and training components
        logging.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, criterion = create_model(num_classes=args.num_classes)
        model.criterion = criterion  # Assign criterion to model
        model = model.to(device)
        
        # Sequential training on each dataset
        current_epoch = 0
        
        # Training sequence: RID → Roofline → AIRS
        dataset_sequence = [
            (args.rid_dir, 'rid'),
            (args.roofline_dir, 'roofline'),
            (args.airs_dir, 'airs')
        ]
        
        for dataset_path, dataset_type in dataset_sequence:
            if dataset_path:
                current_epoch = train_on_dataset(
                    model,
                    dataset_path,
                    dataset_type,
                    args,
                    device,
                    current_epoch
                )
                
                # Save intermediate model after each dataset
                checkpoint = {
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'dataset_type': dataset_type,
                    'num_classes': args.num_classes
                }
                checkpoint_path = os.path.join(args.model_dir, f'model_after_{dataset_type}.pth')
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved model after {dataset_type} training")
        
        # Save final model
        final_checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'num_classes': args.num_classes,
            'training_sequence': [d[1] for d in dataset_sequence if d[0]]
        }
        torch.save(final_checkpoint, os.path.join(args.model_dir, 'final_model.pth'))
        
        logging.info(f"\nTraining completed!")
        logging.info(f"Model saved to: {args.model_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    args = parse_args()
    train(args)