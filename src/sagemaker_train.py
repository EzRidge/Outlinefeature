import os
import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader

# Import our updated components
from models import create_model
from datasets import create_dataloaders
from result_manager import ResultManager

def parse_args():
    """Parse SageMaker training arguments"""
    parser = argparse.ArgumentParser()
    
    # Data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--rid-dir', type=str, default=os.environ.get('SM_CHANNEL_RID', None))
    parser.add_argument('--roofline-dir', type=str, default=os.environ.get('SM_CHANNEL_ROOFLINE', None))
    
    # Training hyperparameters
    parser.add_argument('--epochs-per-dataset', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=12)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--datasets', type=str, default='rid,roofline',
                      help='Comma-separated list of datasets to train on (rid,roofline)')
    parser.add_argument('--test-run', type=bool, default=False,
                      help='If True, only run a few iterations to test setup')
    
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

def train_epoch(model, train_loader, optimizer, device, dataset_type, args):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    # For test runs, only process a few batches
    max_batches = 5 if args.test_run else num_batches
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
            
        # Move data to device
        images = images.to(device)
        device_targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in targets.items()
        }
        
        # Forward pass
        predictions = model(images)
        loss, losses_dict = model.criterion(predictions, device_targets)
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Update total loss (use unscaled loss for logging)
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logging.info(f'Train Batch [{batch_idx + 1}/{num_batches}] - Loss: {loss.item() * args.gradient_accumulation_steps:.4f}')
            for k, v in losses_dict.items():
                logging.info(f'  {k}: {v.item():.4f}')
    
    # Handle any remaining gradients
    if (batch_idx + 1) % args.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / max_batches

def validate(model, val_loader, device, dataset_type, args):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    # For test runs, only process a few batches
    max_batches = 5 if args.test_run else num_batches
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            # Move data to device
            images = images.to(device)
            device_targets = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in targets.items()
            }
            
            # Forward pass
            predictions = model(images)
            loss, _ = model.criterion(predictions, device_targets)
            
            # Update total loss
            total_loss += loss.item()
    
    return total_loss / max_batches

def save_checkpoint(model, optimizer, epoch, dataset_type, val_loss, checkpoint_dir):
    """Save a checkpoint with all necessary info for resuming training"""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'dataset_type': dataset_type
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{dataset_type}_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # If this is the best validation loss, save as best model
    best_model_path = os.path.join(checkpoint_dir, f'best_model_{dataset_type}.pth')
    if not os.path.exists(best_model_path) or val_loss < checkpoint.get('val_loss', float('inf')):
        torch.save(checkpoint, best_model_path)
        logging.info(f"Saved best model for {dataset_type}")

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
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )
    
    # Initialize optimizer with adjusted learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    
    # Training loop for this dataset
    best_val_loss = float('inf')
    epochs = 2 if args.test_run else args.epochs_per_dataset
    
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch
        logging.info(f"\nEpoch [{epoch + 1}/{epochs}] ({dataset_type})")
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, device, dataset_type, args)
        val_loss = validate(model, val_loader, device, dataset_type, args)
        
        logging.info(f"Training Loss: {train_loss:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, current_epoch, dataset_type, 
            val_loss, os.path.join(args.model_dir, 'checkpoints')
        )
    
    return current_epoch + 1

def train(args):
    """Sequential training on multiple datasets"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # Parse datasets to train on
        datasets_to_train = [d.strip().lower() for d in args.datasets.split(',')]
        logging.info(f"Training on datasets: {datasets_to_train}")
        
        # Log all input directories and settings
        logging.info("Input directories:")
        logging.info(f"RID dir: {args.rid_dir}")
        logging.info(f"Roofline dir: {args.roofline_dir}")
        logging.info("\nTraining settings:")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logging.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        logging.info(f"Test run: {args.test_run}")
        logging.info(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
        logging.info(f"Image size: 512x512")
        logging.info(f"Datasets: {datasets_to_train}")
        
        # Create checkpoint directory
        os.makedirs(os.path.join(args.model_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize result manager
        result_manager = ResultManager(os.path.join(args.model_dir, 'results'))
        
        # Initialize model and training components
        logging.info("\nInitializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, criterion = create_model(num_classes=args.num_classes)
        model.criterion = criterion  # Assign criterion to model
        model = model.to(device)
        
        # Sequential training on each dataset
        current_epoch = 0
        
        # Training sequence based on selected datasets
        dataset_sequence = []
        if 'rid' in datasets_to_train and args.rid_dir:
            dataset_sequence.append((args.rid_dir, 'rid'))
        if 'roofline' in datasets_to_train and args.roofline_dir:
            dataset_sequence.append((args.roofline_dir, 'roofline'))
        
        for dataset_path, dataset_type in dataset_sequence:
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
            'training_sequence': [d[1] for d in dataset_sequence]
        }
        torch.save(final_checkpoint, os.path.join(args.model_dir, 'final_model.pth'))
        
        logging.info(f"\nTraining completed!")
        logging.info(f"Model saved to: {args.model_dir}")
        logging.info(f"Trained on datasets: {[d[1] for d in dataset_sequence]}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    args = parse_args()
    train(args)