import torch
import numpy as np

def examine_weights(weights_path):
    print(f"Examining weights file: {weights_path}")
    try:
        # Try loading with different options
        print("\nAttempting to load with default settings...")
        data = torch.load(weights_path, map_location='cpu', weights_only=False)
        print(f"Type of loaded data: {type(data)}")
        if isinstance(data, dict):
            print("\nKeys in state dict:")
            for k in data.keys():
                print(f"  {k}: {data[k].shape if torch.is_tensor(data[k]) else type(data[k])}")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        
    try:
        print("\nAttempting to load as raw bytes...")
        with open(weights_path, 'rb') as f:
            header = f.read(10)
            print(f"First 10 bytes: {[hex(b) for b in header]}")
    except Exception as e:
        print(f"Error reading raw bytes: {str(e)}")

if __name__ == "__main__":
    weights_path = "models/pretrained/roofline_model.pth"
    examine_weights(weights_path)