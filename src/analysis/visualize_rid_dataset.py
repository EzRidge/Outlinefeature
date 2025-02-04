import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = 'Reference Materials/data/RID/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def read_geotiff(filepath):
    """Read a GeoTIFF file and return as numpy array"""
    with rasterio.open(filepath) as src:
        return src.read()

def read_png_mask(filepath):
    """Read a PNG mask file and return as numpy array"""
    mask = np.array(Image.open(filepath))
    # Add channel dimension if needed
    if len(mask.shape) == 2:
        mask = mask[np.newaxis, :, :]
    return mask

def create_azimuth_colormap():
    """Create a colormap for azimuth classes"""
    colors = plt.cm.hsv(np.linspace(0, 1, 16))  # 16 azimuth classes
    colors = np.vstack(([0, 0, 0, 1], colors, [0.7, 0.7, 0.7, 1]))  # Add black for background and gray for flat
    return LinearSegmentedColormap.from_list('azimuth', colors)

def create_superstructure_colormap():
    """Create a colormap for superstructure classes"""
    # Colors for different superstructure types
    colors = [
        [0, 0, 0, 1],        # background
        [1, 0, 0, 1],        # pvmodule
        [0, 1, 0, 1],        # dormer
        [0, 0, 1, 1],        # window
        [1, 1, 0, 1],        # ladder
        [1, 0, 1, 1],        # chimney
        [0, 1, 1, 1],        # shadow
        [0.5, 0.5, 0.5, 1],  # tree
        [0.7, 0.7, 0.7, 1],  # unknown
    ]
    return LinearSegmentedColormap.from_list('superstructures', colors)

def visualize_sample(image_path, segment_mask_path, superstructure_mask_path, output_dir, index):
    """Create visualization for one sample"""
    try:
        # Read data
        image = read_geotiff(image_path)
        segment_mask = read_png_mask(segment_mask_path)
        superstructure_mask = read_png_mask(superstructure_mask_path)
        
        # Convert image to RGB (0-255)
        image = np.transpose(image, (1, 2, 0))  # CHW to HWC
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sample {index}')
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot segment mask
        segment_plot = axes[1].imshow(segment_mask[0], cmap=create_azimuth_colormap())
        axes[1].set_title('Segment Mask\n(colors represent azimuth)')
        axes[1].axis('off')
        plt.colorbar(segment_plot, ax=axes[1])
        
        # Plot superstructure mask
        superstructure_plot = axes[2].imshow(superstructure_mask[0], cmap=create_superstructure_colormap())
        axes[2].set_title('Superstructure Mask')
        axes[2].axis('off')
        plt.colorbar(superstructure_plot, ax=axes[2])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{index}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error processing sample {index}: {str(e)}")
        return False

def create_dataset_info(output_dir):
    """Create dataset information file"""
    info_text = """RID Dataset Information
======================

Dataset Overview:
----------------
- 1,880 roof-centered images
- GeoTIFF format from Google Maps Static API
- Two types of masks:
  1. Segment masks (18 classes):
     * Background (0)
     * 16 azimuth classes (1-16)
     * Flat (17)
  2. Superstructure masks:
     * Background (0)
     * PV Module
     * Dormer
     * Window
     * Ladder
     * Chimney
     * Shadow
     * Tree
     * Unknown

Visualization Files:
------------------
sample_[0-4].png: Example visualizations showing:
- Left: Original aerial image
- Middle: Segment mask (colors represent azimuth angles)
- Right: Superstructure mask (different colors for each type)

Note: Images are georeferenced to enable precise mask creation from annotation data.
"""
    
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(info_text)

def main():
    # Create output directory
    output_dir = create_output_dir()
    
    # Base paths
    base_dir = 'Reference Materials/data/RID/m1655470/RID_dataset'
    images_dir = os.path.join(base_dir, 'images_roof_centered_geotiff')
    segments_dir = os.path.join(base_dir, 'masks_segments_reviewed')
    superstructures_dir = os.path.join(base_dir, 'masks_superstructures_reviewed')
    
    # Get first 5 samples
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])[:5]
    
    print("Creating visualizations...")
    successful = 0
    for i, image_file in enumerate(image_files):
        base_name = image_file.replace('.tif', '')
        
        image_path = os.path.join(images_dir, image_file)
        segment_path = os.path.join(segments_dir, f"{base_name}.png")
        superstructure_path = os.path.join(superstructures_dir, f"{base_name}.png")
        
        print(f"Processing sample {i}: {base_name}")
        if visualize_sample(image_path, segment_path, superstructure_path, output_dir, i):
            successful += 1
    
    # Create dataset info
    create_dataset_info(output_dir)
    print(f"\nVisualization complete! Successfully processed {successful} samples.")
    print("Check the output directory:")
    print(output_dir)

if __name__ == "__main__":
    main()
