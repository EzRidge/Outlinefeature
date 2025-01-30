"""
Inference script for roof feature detection and measurement.
"""

import os
import json
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image

from .models import MultiScaleCNN
from .utils import calculate_measurements

class RoofPredictor:
    """
    Class for running inference on new roof images.
    """
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to trained model checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = MultiScaleCNN()
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def preprocess_image(self, image_path, target_size=(1024, 1024)):
        """
        Preprocess image for inference.
        
        Args:
            image_path (str): Path to input image
            target_size (tuple): Target size for resizing
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Read and resize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        image = image.unsqueeze(0)  # Add batch dimension
        
        # Normalize with ImageNet stats
        normalize = torch.nn.functional.normalize
        image = normalize(image, 
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        
        return image
    
    def predict_masks(self, image):
        """
        Run inference to get segmentation masks.
        
        Args:
            image (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: Predicted masks for (roof, ridge lines, eave lines)
        """
        with torch.no_grad():
            # Move image to device and run inference
            image = image.to(self.device)
            output = self.model(image)
            
            # Get predictions
            pred = torch.argmax(output, dim=1)
            pred = pred.cpu().numpy()[0]  # Remove batch dimension
            
            # Split into separate masks
            roof_mask = (pred == 1).astype(np.uint8)
            ridge_mask = (pred == 2).astype(np.uint8)
            eave_mask = (pred == 3).astype(np.uint8)
            
            return roof_mask, ridge_mask, eave_mask
    
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image and save results.
        
        Args:
            image_path (str): Path to input image
            output_dir (str, optional): Directory to save results
            
        Returns:
            dict: Dictionary containing measurements and file paths
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Get predictions
        roof_mask, ridge_mask, eave_mask = self.predict_masks(image)
        
        # Calculate measurements
        measurements = calculate_measurements(
            roof_mask, ridge_mask, eave_mask
        )
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Save masks
            base_name = Path(image_path).stem
            cv2.imwrite(
                str(output_dir / f'{base_name}_roof_mask.png'),
                roof_mask * 255
            )
            cv2.imwrite(
                str(output_dir / f'{base_name}_ridge_mask.png'),
                ridge_mask * 255
            )
            cv2.imwrite(
                str(output_dir / f'{base_name}_eave_mask.png'),
                eave_mask * 255
            )
            
            # Save measurements
            with open(output_dir / f'{base_name}_measurements.json', 'w') as f:
                json.dump(measurements, f, indent=2)
        
        return {
            'measurements': measurements,
            'roof_mask': roof_mask,
            'ridge_mask': ridge_mask,
            'eave_mask': eave_mask
        }
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save results
            
        Returns:
            list: List of dictionaries containing results for each image
        """
        results = []
        input_dir = Path(input_dir)
        
        # Process each image
        for img_path in input_dir.glob('*.jpg'):
            try:
                result = self.process_image(str(img_path), output_dir)
                results.append({
                    'image_path': str(img_path),
                    **result
                })
            except Exception as e:
                print(f'Error processing {img_path}: {str(e)}')
        
        return results

def main(model_path, input_path, output_dir, device='cuda'):
    """
    Main function to run inference.
    
    Args:
        model_path (str): Path to trained model checkpoint
        input_path (str): Path to input image or directory
        output_dir (str): Directory to save results
        device (str): Device to run inference on
    """
    # Initialize predictor
    predictor = RoofPredictor(model_path, device)
    
    # Process input
    input_path = Path(input_path)
    if input_path.is_file():
        result = predictor.process_image(str(input_path), output_dir)
        print(f'Processed {input_path}')
        print('Measurements:', json.dumps(result['measurements'], indent=2))
    else:
        results = predictor.process_directory(str(input_path), output_dir)
        print(f'Processed {len(results)} images')
        
        # Save summary
        summary_path = Path(output_dir) / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {summary_path}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run roof feature detection')
    parser.add_argument('model_path', help='Path to trained model checkpoint')
    parser.add_argument('input_path', help='Path to input image or directory')
    parser.add_argument('output_dir', help='Directory to save results')
    parser.add_argument('--device', default='cuda', help='Device to run inference on')
    
    args = parser.parse_args()
    main(args.model_path, args.input_path, args.output_dir, args.device)
