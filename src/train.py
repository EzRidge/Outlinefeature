"""
Training script for the hybrid roof feature detection model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm

from .models import create_model
from .config import MODEL_CONFIG, TRAIN_CONFIG, LOGGING_CONFIG
from .data import RoofDataset

def setup_logging():
    """Configure logging for training."""
    log_dir = LOGGING_CONFIG['log_dir']
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs):
    """Train the model."""
    device = torch.device(TRAIN_CONFIG['device'])
    model = model.to(device)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG['early_stopping']['patience'],
        min_delta=TRAIN_CONFIG['early_stopping']['min_delta']
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Calculate loss for each output head
                loss = 0
                for key in ['outline', 'ridge', 'hip', 'valley']:
                    loss += criterion(outputs[key], targets[key])
                
                # Add angle prediction loss if available
                if 'angles' in outputs and 'angles' in targets:
                    angle_loss = criterion(outputs['angles'], targets['angles'])
                    loss += TRAIN_CONFIG['loss_weights']['angle'] * angle_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log batch metrics
                if batch_idx % LOGGING_CONFIG['log_frequency'] == 0:
                    logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                outputs = model(images)
                
                # Calculate validation loss
                batch_loss = 0
                for key in ['outline', 'ridge', 'hip', 'valley']:
                    batch_loss += criterion(outputs[key], targets[key])
                
                if 'angles' in outputs and 'angles' in targets:
                    angle_loss = criterion(outputs['angles'], targets['angles'])
                    batch_loss += TRAIN_CONFIG['loss_weights']['angle'] * angle_loss
                
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch metrics
        logging.info(f'Epoch {epoch+1}:')
        logging.info(f'  Training Loss: {avg_train_loss:.4f}')
        logging.info(f'  Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = Path(TRAIN_CONFIG['model_save_path']) / 'best_model.pth'
            torch.save(model.state_dict(), model_path)
            logging.info(f'  Saved new best model with validation loss: {best_val_loss:.4f}')
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            logging.info('Early stopping triggered')
            break
        
        # Save checkpoint
        if (epoch + 1) % TRAIN_CONFIG['save_frequency'] == 0:
            checkpoint_path = Path(TRAIN_CONFIG['model_save_path']) / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            logging.info(f'  Saved checkpoint at epoch {epoch+1}')

def main():
    """Main training function."""
    setup_logging()
    logging.info('Starting training')
    
    # Create model
    model = create_model(MODEL_CONFIG)
    logging.info(f'Created model: {model.__class__.__name__}')
    
    # Setup data loaders
    train_dataset = RoofDataset(split='train')
    val_dataset = RoofDataset(split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    
    # Train model
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=MODEL_CONFIG['epochs']
    )
    
    logging.info('Training completed')

if __name__ == '__main__':
    main()
