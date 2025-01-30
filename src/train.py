"""
Training utilities for the roof feature detection model.
"""

import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import MultiScaleCNN
from .data import create_dataloaders

class Trainer:
    """
    Trainer class to handle model training and validation.
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 device='cuda',
                 save_dir='models'):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimization algorithm
            device: Device to train on ('cuda' or 'cpu')
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for images, masks in pbar:
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for images, masks in pbar:
                    # Move data to device
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                    # Update progress bar
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, save_freq=5):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_freq (int): Frequency of saving model checkpoints
        """
        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Log metrics
            self.logger.info(
                f'Train Loss: {train_loss:.4f} | '
                f'Val Loss: {val_loss:.4f}'
            )
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model('best_model.pth')
                self.logger.info('Saved best model')
            
            # Save periodic checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Log training completion
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time/60:.2f} minutes')
    
    def save_model(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_model(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_model(image_dir, 
                mask_dir, 
                num_epochs=50,
                batch_size=8,
                learning_rate=1e-4,
                device='cuda'):
    """
    Convenience function to train the model.
    
    Args:
        image_dir (str): Directory containing training images
        mask_dir (str): Directory containing mask annotations
        num_epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization
        device (str): Device to train on ('cuda' or 'cpu')
    """
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        image_dir, 
        mask_dir,
        batch_size=batch_size
    )
    
    # Initialize model and training components
    model = MultiScaleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    trainer.train(num_epochs)
    return trainer
