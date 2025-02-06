import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging

class ResultManager:
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_dir, 'hybrid_model', self.timestamp)
        
        # Create directories
        for subdir in ['checkpoints', 'visualizations', 'metrics', 'report', 'logs']:
            os.makedirs(os.path.join(self.run_dir, subdir), exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize metrics file
        self.init_metrics_file()
        
        # Define custom colormap for segments matching notebook
        self.segment_colors = {
            0: [1.0, 1.0, 0.0],    # Background: Yellow
            1: [0.5, 0.0, 0.5],    # PV module: Purple
            2: [0.0, 0.8, 0.8],    # Dormer: Teal
            3: [0.8, 0.0, 0.0],    # Window: Dark Red
            4: [0.0, 0.6, 0.0],    # Ladder: Dark Green
            5: [0.0, 0.0, 0.8],    # Chimney: Dark Blue
            6: [0.4, 0.4, 0.4],    # Shadow: Gray
            7: [0.8, 0.4, 0.0],    # Tree: Brown
            8: [0.7, 0.7, 0.7]     # Unknown: Light Gray
        }
        
        # Define colors for line types
        self.line_colors = {
            'ridge': [1.0, 0.0, 0.0],    # Bright Red
            'hip': [0.0, 0.0, 1.0],      # Bright Blue
            'valley': [0.0, 1.0, 0.0]    # Bright Green
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.run_dir, 'logs', 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_metrics_file(self):
        """Initialize metrics CSV file with headers"""
        metrics_file = os.path.join(self.run_dir, 'metrics', 'losses.csv')
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'segments_loss', 
                           'superstructures_loss', 'lines_loss', 'depth_loss'])
    
    def _apply_segment_colormap(self, segment_mask):
        """Convert segment indices to RGB colors"""
        h, w = segment_mask.shape
        rgb_mask = np.zeros((h, w, 3))
        for idx, color in self.segment_colors.items():
            mask = (segment_mask == idx)
            rgb_mask[mask] = color
        return rgb_mask
    
    def _create_line_visualization(self, line_pred, color, threshold=0.5):
        """Create visualization for a single line type"""
        # Convert to binary mask with threshold
        binary_mask = (line_pred > threshold).astype(float)
        
        # Create RGB image with black background
        rgb_img = np.zeros((*line_pred.shape, 3))
        
        # Only set color for pixels above threshold
        rgb_img[binary_mask > 0] = color
        
        return rgb_img
    
    def save_visualization(self, epoch, image, predictions, targets, sample_idx=0):
        """Save visualization for current epoch"""
        try:
            viz_dir = os.path.join(self.run_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create figure with subplots (2 rows, 4 columns like the notebook)
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'Epoch {epoch} - Sample {sample_idx}', y=1.02)
            
            # Row 1: Original, Segments, Superstructures, Depth
            # Original image
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            axes[0,0].imshow(img_np)
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')
            
            # Segment prediction
            seg_pred = predictions['segments'][0].argmax(dim=0).cpu().numpy()
            seg_pred_rgb = self._apply_segment_colormap(seg_pred)
            axes[0,1].imshow(seg_pred_rgb)
            axes[0,1].set_title('Segments')
            axes[0,1].axis('off')
            
            # Superstructure prediction (binary)
            super_pred = torch.sigmoid(predictions['superstructures'][0, 0]).cpu().numpy()
            axes[0,2].imshow(super_pred > 0.5, cmap='binary')
            axes[0,2].set_title('Superstructures')
            axes[0,2].axis('off')
            
            # Depth prediction
            depth_pred = predictions['depth'][0, 0].cpu().numpy()
            depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min())
            axes[0,3].imshow(depth_pred, cmap='viridis')
            axes[0,3].set_title('Depth')
            axes[0,3].axis('off')
            
            # Row 2: Line Predictions
            # Ridge Lines (red)
            ridge_pred = torch.sigmoid(predictions['lines'][0, 0]).cpu().numpy()
            ridge_viz = self._create_line_visualization(ridge_pred, self.line_colors['ridge'])
            axes[1,0].imshow(ridge_viz)
            axes[1,0].set_title('Ridge Lines')
            axes[1,0].axis('off')
            
            # Hip Lines (blue)
            hip_pred = torch.sigmoid(predictions['lines'][0, 1]).cpu().numpy()
            hip_viz = self._create_line_visualization(hip_pred, self.line_colors['hip'])
            axes[1,1].imshow(hip_viz)
            axes[1,1].set_title('Hip Lines')
            axes[1,1].axis('off')
            
            # Valley Lines (green)
            valley_pred = torch.sigmoid(predictions['lines'][0, 2]).cpu().numpy()
            valley_viz = self._create_line_visualization(valley_pred, self.line_colors['valley'])
            axes[1,2].imshow(valley_viz)
            axes[1,2].set_title('Valley Lines')
            axes[1,2].axis('off')
            
            # Combined Lines
            combined_lines = np.zeros((*ridge_pred.shape, 3))
            ridge_mask = ridge_pred > 0.5
            hip_mask = hip_pred > 0.5
            valley_mask = valley_pred > 0.5
            combined_lines[ridge_mask] = self.line_colors['ridge']
            combined_lines[hip_mask] = self.line_colors['hip']
            combined_lines[valley_mask] = self.line_colors['valley']
            axes[1,3].imshow(combined_lines)
            axes[1,3].set_title('Combined Lines')
            axes[1,3].axis('off')
            
            # Adjust spacing
            plt.tight_layout()
            
            # Save visualization
            save_path = os.path.join(viz_dir, f'epoch_{epoch:03d}_sample_{sample_idx:02d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved visualization for epoch {epoch} sample {sample_idx}")
            
        except Exception as e:
            self.logger.error(f"Error saving visualization: {str(e)}")
            plt.close()
    
    def save_metrics(self, epoch, train_metrics, val_metrics=None):
        """Save metrics to CSV"""
        try:
            metrics_file = os.path.join(self.run_dir, 'metrics', 'losses.csv')
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [epoch, train_metrics['total']]
                if val_metrics:
                    row.append(val_metrics['total'])
                else:
                    row.append('')
                
                # Add individual losses
                for loss_type in ['segments', 'superstructures', 'lines', 'depth']:
                    row.append(train_metrics.get(loss_type, ''))
                
                writer.writerow(row)
            
            self.logger.info(f"Saved metrics for epoch {epoch}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
    
    def save_checkpoint(self, model, optimizer, epoch, loss, is_best=False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            
            # Save latest
            latest_path = os.path.join(self.run_dir, 'checkpoints', 'latest.pth')
            torch.save(checkpoint, latest_path)
            
            # Save best if needed
            if is_best:
                best_path = os.path.join(self.run_dir, 'checkpoints', 'best.pth')
                torch.save(checkpoint, best_path)
                self.logger.info(f"Saved best model checkpoint at epoch {epoch}")
            
            self.logger.info(f"Saved checkpoint for epoch {epoch}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
    
    def create_html_report(self):
        """Generate final HTML report"""
        try:
            report_path = os.path.join(self.run_dir, 'report', 'index.html')
            
            # Basic HTML template
            html_content = f"""
            <html>
            <head>
                <title>Training Report - {self.timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .visualization {{ margin: 20px 0; text-align: center; }}
                    img {{ max-width: 100%; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .epoch-section {{ margin-bottom: 40px; }}
                    h1, h2, h3 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>Training Report - {self.timestamp}</h1>
                
                <h2>Training Progress</h2>
                <div class="visualization">
                    <img src="../metrics/loss_plot.png" alt="Loss Plot">
                </div>
                
                <h2>Sample Predictions</h2>
                {self._get_visualization_html()}
                
                <h2>Training Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Timestamp</td><td>{self.timestamp}</td></tr>
                </table>
            </body>
            </html>
            """
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info("Created HTML report")
        except Exception as e:
            self.logger.error(f"Error creating HTML report: {str(e)}")
    
    def _get_visualization_html(self):
        """Generate HTML for visualizations"""
        try:
            viz_dir = os.path.join(self.run_dir, 'visualizations')
            html = []
            
            # Get all visualization files
            viz_files = []
            for file in os.listdir(viz_dir):
                if file.endswith('.png'):
                    epoch = int(file.split('_')[1])
                    viz_files.append((epoch, file))
            
            # Sort by epoch
            viz_files.sort(key=lambda x: x[0])
            
            # Group by epoch
            current_epoch = None
            for epoch, file in viz_files:
                if epoch != current_epoch:
                    if current_epoch is not None:
                        html.append('</div>')
                    current_epoch = epoch
                    html.append(f'<div class="epoch-section">')
                    html.append(f'<h3>Epoch {epoch}</h3>')
                html.append(f'<img src="../visualizations/{file}">')
            
            if viz_files:
                html.append('</div>')
            
            return '\n'.join(html)
        except Exception as e:
            self.logger.error(f"Error generating visualization HTML: {str(e)}")
            return "<p>Error loading visualizations</p>"
    
    def plot_losses(self):
        """Plot training and validation losses"""
        try:
            metrics_file = os.path.join(self.run_dir, 'metrics', 'losses.csv')
            epochs, train_losses, val_losses = [], [], []
            
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    train_losses.append(float(row['train_loss']))
                    if row['val_loss']:
                        val_losses.append(float(row['val_loss']))
            
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
            if val_losses:
                plt.plot(epochs, val_losses, label='Val Loss', color='red', 
                        linewidth=2, marker='o', markersize=6)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add minor gridlines
            plt.grid(True, which='minor', linestyle=':', alpha=0.4)
            plt.minorticks_on()
            
            # Save plot
            plt.savefig(os.path.join(self.run_dir, 'metrics', 'loss_plot.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Generated loss plot")
        except Exception as e:
            self.logger.error(f"Error plotting losses: {str(e)}")
