import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
output_dir = 'Reference Materials/data/Roofline-Extraction/visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load .mat file
mat_path = 'Reference Materials/data/Roofline-Extraction/imdb.mat'
data = h5py.File(mat_path, 'r')

# Get data
images = data['imdb/images']
depths = data['imdb/depths']
elements = data['imdb/elements']

# Function to display and save one sample
def save_sample(idx=0):
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Sample {idx}')
    
    # Original image
    img = np.transpose(images[idx], (1, 2, 0))  # Change from (3,224,224) to (224,224,3)
    axes[0,0].imshow(img.astype(np.uint8))
    axes[0,0].set_title('Original Image')
    
    # Depth map
    depth_plot = axes[0,1].imshow(depths[idx], cmap='viridis')
    axes[0,1].set_title('Depth Map')
    plt.colorbar(depth_plot, ax=axes[0,1])
    
    # Elements (3 channels)
    element_names = ['Ridge Lines', 'Hip Lines', 'Valley Lines']
    for i in range(3):
        axes[1,i].imshow(elements[idx,i], cmap='gray')
        axes[1,i].set_title(f'{element_names[i]}')
    
    # Hide empty subplot
    axes[0,2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'sample_{idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Save first 5 samples
print("Saving sample visualizations...")
for i in range(5):
    print(f"Processing sample {i}")
    save_sample(i)

# Save dataset information
print("\nSaving dataset information...")
info_text = f"""Roofline-Extraction Dataset Information
================================

Dataset Structure:
-----------------
Number of samples: {images.shape[0]}
Image resolution: {images.shape[2]}x{images.shape[3]}
Number of channels: {images.shape[1]} (RGB)

Components:
----------
1. Images:
   - Shape: {images.shape}
   - Type: RGB aerial images
   - Value range: 0-255

2. Depth Maps:
   - Shape: {depths.shape}
   - Type: Single-channel depth information
   - Contains height/elevation data

3. Element Maps:
   - Shape: {elements.shape}
   - Type: Binary masks for roof elements
   - Channels:
     * Channel 0: Ridge lines
     * Channel 1: Hip lines
     * Channel 2: Valley lines

Visualization Files:
------------------
- sample_0.png to sample_4.png: Example visualizations showing:
  * Original aerial image
  * Depth map (with color scale)
  * Three element channels (ridge, hip, valley lines)
"""

with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
    f.write(info_text)

print("\nVisualization complete! Check the output directory:")
print(output_dir)

# Close the file
data.close()
