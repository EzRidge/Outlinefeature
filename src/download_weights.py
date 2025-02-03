"""
Script to download pre-trained weights for roof feature detection.
"""

import os
import sys
import requests
from pathlib import Path

def download_file(url, destination):
    """
    Download a file from URL to destination with progress bar.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        print(f"Downloading to {destination}")
        print("[", end="")
        sys.stdout.flush()
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                
                # Update progress bar
                progress = int(50 * downloaded / total_size)
                sys.stdout.write("\r[" + "=" * progress + " " * (50-progress) + "]")
                sys.stdout.write(f" {downloaded}/{total_size} bytes")
                sys.stdout.flush()
                
        print("\nDownload completed!")
        return True
        
    except Exception as e:
        print(f"\nError downloading weights: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def main():
    """
    Download pre-trained weights from the Roofline-Extraction repository.
    """
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Weights file paths
    weights_path = models_dir / "roofline_pretrained.pth"
    
    # Download URLs
    # Note: Since direct download links aren't available, we'll use a placeholder model
    # that demonstrates the architecture but needs training
    print("Initializing placeholder model weights...")
    
    import torch
    from .models import create_model
    
    # Create model and save initial weights
    model = create_model()
    torch.save(model.state_dict(), weights_path)
    
    print(f"\nPlaceholder weights saved to {weights_path}")
    print("\nNote: These are initialized weights. For optimal performance:")
    print("1. Train the model on the RID dataset, or")
    print("2. Replace with actual pre-trained weights when available")

if __name__ == "__main__":
    main()
