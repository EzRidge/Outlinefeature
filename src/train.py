import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from src.result_manager import ResultManager

def train_model(model, train_loader, val_loader, criterion, optimizer, config, result_manager):
    """Train the model with comprehensive logging and visualization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        task_losses = {'segments': 0, 'superstructures': 0, 'lines': 0, 'depth': 0}
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move to device
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss, losses = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            for k, v in losses.items():
                task_losses[k] += v.item()
            
            if batch_idx % 10 == 0:
                logging.info(f'Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        # Average training losses
        num_batches = len(train_loader)
        train_loss /= num_batches
        task_losses = {k: v/num_batches for k, v in task_losses.items()}
        task_losses['total'] = train_loss
        
        # Validation
        if (epoch + 1) % config.val_interval == 0:
            model.eval()
            val_loss = 0
            val_task_losses = {'segments': 0, 'superstructures': 0, 'lines': 0, 'depth': 0}
            
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_loader):
                    images = images.to(device)
                    targets = {k: v.to(device) for k, v in targets.items()}
                    
                    predictions = model(images)
                    loss, losses = criterion(predictions, targets)
                    
                    val_loss += loss.item()
                    for k, v in losses.items():
                        val_task_losses[k] += v.item()
                    
                    # Save visualization for first batch
                    if batch_idx == 0:
                        result_manager.save_visualization(
                            epoch + 1,
                            images[0],
                            predictions,
                            targets
                        )
            
            # Average validation losses
            num_val_batches = len(val_loader)
            val_loss /= num_val_batches
            val_task_losses = {k: v/num_val_batches for k, v in val_task_losses.items()}
            val_task_losses['total'] = val_loss
            
            # Save metrics
            result_manager.save_metrics(epoch + 1, task_losses, val_task_losses)
            
            # Update best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            result_manager.save_checkpoint(model, optimizer, epoch + 1, val_loss, is_best)
            
            # Log progress
            logging.info(f"\nEpoch {epoch+1}")
            logging.info(f"Training Loss: {train_loss:.4f}")
            logging.info(f"Validation Loss: {val_loss:.4f}")
            logging.info("\nTask Losses:")
            for k in task_losses.keys():
                if k != 'total':
                    logging.info(f"{k:>15}: Train={task_losses[k]:.4f}, Val={val_task_losses[k]:.4f}")
        else:
            # Save metrics for training only
            result_manager.save_metrics(epoch + 1, task_losses)
    
    # Final tasks
    result_manager.plot_losses()
    result_manager.create_html_report()
    
    return model, best_val_loss

def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['loss']
