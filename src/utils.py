"""
Utility functions for roof feature detection.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from .config import MODEL_CONFIG, OUTPUT_DIR

def visualize_predictions(image, predictions, save_path=None):
    """
    Visualize model predictions.
    
    Args:
        image: RGB image tensor [C, H, W]
        predictions: Dict of prediction tensors
        save_path: Optional path to save visualization
    """
    # Convert image tensor to numpy
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    
    # Outline prediction
    outline = predictions['outline'].cpu().numpy()[0]
    axes[1].imshow(outline, cmap='jet')
    axes[1].set_title('Roof Outline')
    
    # Ridge lines
    ridge = predictions['ridge'].cpu().numpy()[0]
    axes[2].imshow(ridge, cmap='jet')
    axes[2].set_title('Ridge Lines')
    
    # Hip lines
    hip = predictions['hip'].cpu().numpy()[0]
    axes[3].imshow(hip, cmap='jet')
    axes[3].set_title('Hip Lines')
    
    # Valley lines
    valley = predictions['valley'].cpu().numpy()[0]
    axes[4].imshow(valley, cmap='jet')
    axes[4].set_title('Valley Lines')
    
    # Angles
    angles = predictions['angles'].cpu().numpy()[0]
    im = axes[5].imshow(angles, cmap='hsv')
    axes[5].set_title('Angle Predictions')
    plt.colorbar(im, ax=axes[5])
    
    # Adjust layout and save/show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def overlay_features(image, predictions, threshold=0.5):
    """
    Overlay predicted features on the original image.
    
    Args:
        image: RGB image tensor [C, H, W]
        predictions: Dict of prediction tensors
        threshold: Confidence threshold for predictions
    
    Returns:
        Annotated image with overlaid features
    """
    # Convert image tensor to numpy
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    result = image.copy()
    
    # Overlay outline
    outline = (predictions['outline'].cpu().numpy()[0] > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Overlay ridge lines
    ridge = (predictions['ridge'].cpu().numpy()[0] > threshold).astype(np.uint8)
    ridge_points = np.where(ridge > 0)
    result[ridge_points[0], ridge_points[1]] = [255, 0, 0]  # Red
    
    # Overlay hip lines
    hip = (predictions['hip'].cpu().numpy()[0] > threshold).astype(np.uint8)
    hip_points = np.where(hip > 0)
    result[hip_points[0], hip_points[1]] = [0, 0, 255]  # Blue
    
    # Overlay valley lines
    valley = (predictions['valley'].cpu().numpy()[0] > threshold).astype(np.uint8)
    valley_points = np.where(valley > 0)
    result[valley_points[0], valley_points[1]] = [255, 255, 0]  # Yellow
    
    return result

def save_predictions(image, predictions, filename, output_dir=OUTPUT_DIR):
    """
    Save model predictions and visualizations.
    
    Args:
        image: RGB image tensor
        predictions: Dict of prediction tensors
        filename: Base filename for saving
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual feature maps
    for key, pred in predictions.items():
        pred_np = pred.cpu().numpy()[0]
        np.save(output_dir / f'{filename}_{key}.npy', pred_np)
    
    # Save visualization
    visualize_predictions(
        image, 
        predictions,
        save_path=output_dir / f'{filename}_visualization.png'
    )
    
    # Save overlaid features
    overlay = overlay_features(image, predictions)
    cv2.imwrite(
        str(output_dir / f'{filename}_overlay.png'),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Dict of prediction tensors
        targets: Dict of target tensors
        threshold: Confidence threshold for predictions
    
    Returns:
        Dict of metrics
    """
    metrics = {}
    
    for key in ['outline', 'ridge', 'hip', 'valley']:
        pred = (predictions[key] > threshold).float()
        target = targets[key]
        
        # Calculate IoU
        intersection = (pred * target).sum()
        union = (pred + target).clamp(0, 1).sum()
        iou = (intersection / (union + 1e-6)).item()
        
        # Calculate precision and recall
        true_pos = (pred * target).sum()
        false_pos = (pred * (1 - target)).sum()
        false_neg = ((1 - pred) * target).sum()
        
        precision = (true_pos / (true_pos + false_pos + 1e-6)).item()
        recall = (true_pos / (true_pos + false_neg + 1e-6)).item()
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        metrics[key] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate angle error if available
    if 'angles' in predictions and 'angles' in targets:
        angle_error = torch.abs(predictions['angles'] - targets['angles'])
        metrics['angle_mae'] = angle_error.mean().item()
        metrics['angle_rmse'] = torch.sqrt((angle_error ** 2).mean()).item()
    
    return metrics

def log_metrics(metrics, epoch, split='train'):
    """Log evaluation metrics."""
    logging.info(f'{split.capitalize()} Metrics - Epoch {epoch}:')
    
    for feature in ['outline', 'ridge', 'hip', 'valley']:
        logging.info(f'  {feature.capitalize()}:')
        for metric, value in metrics[feature].items():
            logging.info(f'    {metric}: {value:.4f}')
    
    if 'angle_mae' in metrics:
        logging.info('  Angles:')
        logging.info(f'    MAE: {metrics["angle_mae"]:.4f}')
        logging.info(f'    RMSE: {metrics["angle_rmse"]:.4f}')
