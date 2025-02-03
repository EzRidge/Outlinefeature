"""
Training script for hybrid roof feature detection model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import albumentations as A
from .models import create_model, load_pretrained
from .config import MODEL_CONFIG, DATASET_CONFIG, TRAIN_CONFIG

class RoofDataset(Dataset):
    """Dataset class for roof feature detection."""
    def __init__(self, data_dir, transform=None, phase='train'):
        self.data_dir = Path(data_dir) / phase
        self.transform = transform
        
        # Get all image files
        self.image_files = list((self.data_dir / "images").glob("*.png"))
        print(f"Found {len(self.image_files)} images in {phase} set")
        
        # Setup augmentations
        if phase == 'train' and MODEL_CONFIG['augmentation']:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3)
            ])
        else:
            self.aug = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load masks
        mask_dir = self.data_dir / "masks" / image_path.stem
        masks = {}
        for feature in ['outline', 'ridge', 'hip', 'valley']:
            mask_path = mask_dir / f"{feature}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                masks[feature] = mask / 255.0  # Normalize to [0, 1]
            else:
                masks[feature] = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Apply augmentations
        if self.aug is not None:
            augmented = self.aug(image=image, masks=list(masks.values()))
            image = augmented['image']
            masks = dict(zip(masks.keys(), augmented['masks']))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert masks to tensors
        for k, v in masks.items():
            masks[k] = torch.from_numpy(v).float()
        
        return image, masks

class FeatureLoss(nn.Module):
    """Custom loss function for feature detection."""
    def __init__(self, weights=TRAIN_CONFIG['loss_weights']):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, targets):
        total_loss = 0
        losses = {}
        
        # Segmentation loss for each feature
        for feature in ['outline', 'ridge', 'hip', 'valley']:
            if feature in outputs and feature in targets:
                loss = self.bce(outputs[feature], targets[feature])
                losses[f'{feature}_loss'] = loss
                total_loss += self.weights['segmentation'] * loss
        
        # Boundary loss to encourage sharp edges
        if 'outline' in outputs and 'outline' in targets:
            edge_loss = self.boundary_loss(outputs['outline'], targets['outline'])
            losses['boundary_loss'] = edge_loss
            total_loss += self.weights['boundary'] * edge_loss
        
        # Angle consistency loss
        if 'angles' in outputs:
            angle_loss = self.angle_consistency_loss(outputs)
            losses['angle_loss'] = angle_loss
            total_loss += self.weights['angle'] * angle_loss
        
        return total_loss, losses
    
    def boundary_loss(self, pred, target):
        """Calculate boundary loss using gradient difference."""
        pred_grad = self.gradient_magnitude(pred)
        target_grad = self.gradient_magnitude(target)
        return self.mse(pred_grad, target_grad)
    
    def gradient_magnitude(self, x):
        """Calculate gradient magnitude."""
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def angle_consistency_loss(self, outputs):
        """Encourage consistent angles between ridge, hip, and valley lines."""
        angles = outputs['angles']
        # Calculate angle differences between adjacent pixels
        angle_diff = torch.abs(angles[:, :, 1:, :] - angles[:, :, :-1, :])
        angle_diff = torch.min(angle_diff, 2*np.pi - angle_diff)
        return torch.mean(angle_diff)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model."""
    # Setup logging
    log_dir = TRAIN_CONFIG['log_dir']
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=str(log_file), level=logging.INFO)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            # Move data to device
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(images)
            loss, component_losses = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Log batch results
            if (batch_idx + 1) % TRAIN_CONFIG['log_frequency'] == 0:
                log_msg = f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                for k, v in component_losses.items():
                    log_msg += f", {k}: {v.item():.4f}"
                logging.info(log_msg)
        
        avg_train_loss = np.mean(train_losses)
        print(f"Training Loss: {avg_train_loss:.4f}")
        logging.info(f"Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % TRAIN_CONFIG['save_frequency'] == 0:
            checkpoint_path = MODELS_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
        
        # Early stopping
        if avg_val_loss < best_val_loss - TRAIN_CONFIG['early_stopping']['min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['early_stopping']['patience']:
                print("Early stopping triggered")
                logging.info("Early stopping triggered")
                break

def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description="Train roof feature detection model")
    parser.add_argument("--data_dir", default="data/processed",
                       help="Directory containing processed dataset")
    parser.add_argument("--weights", help="Path to pretrained weights")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Load weights if provided
    if args.weights:
        model = load_pretrained(model, args.weights)
    
    # Create datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RoofDataset(args.data_dir, transform=transform, phase='train')
    val_dataset = RoofDataset(args.data_dir, transform=transform, phase='val')
    
    train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'],
                            shuffle=True, num_workers=TRAIN_CONFIG['num_workers'],
                            pin_memory=TRAIN_CONFIG['pin_memory'])
    
    val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'],
                          shuffle=False, num_workers=TRAIN_CONFIG['num_workers'],
                          pin_memory=TRAIN_CONFIG['pin_memory'])
    
    # Setup loss and optimizer
    criterion = FeatureLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'],
                          weight_decay=MODEL_CONFIG['weight_decay'])
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer,
                MODEL_CONFIG['epochs'] - start_epoch, device)

if __name__ == "__main__":
    main()
