"""
Quick test script for roof feature detection.
"""

import argparse
import cv2
import os
from pathlib import Path
import torch
from .models import create_model, load_pretrained
from .utils import preprocess_image, process_features, visualize_results

def run_quick_test(image_path, output_path=None):
    """
    Run quick test of roof feature detection.
    """
    print(f"\nProcessing image: {image_path}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create output directory if needed
    if output_path is None:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(image_path).stem}_results.png"
    
    try:
        # Create and initialize model
        print("Initializing model...")
        model = create_model()
        model.eval()  # Set to evaluation mode
        
        # Process image
        print("Processing image...")
        features, measurements = process_features(image, model)
        
        # Generate visualization
        print("\nGenerating visualization...")
        visualize_results(image, features, measurements, str(output_path))
        
        print("\nMeasurements:")
        for key, value in sorted(measurements.items()):
            if isinstance(value, float):
                if 'area' in key:
                    print(f"{key}: {value:,.2f} sq ft")
                elif 'length' in key or 'perimeter' in key:
                    print(f"{key}: {value:,.2f} ft")
                else:
                    print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")
        
        print(f"\nResults saved to {output_path.parent}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test roof feature detection")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", help="Path to output visualization")
    args = parser.parse_args()
    
    run_quick_test(args.image_path, args.output)

if __name__ == "__main__":
    main()
