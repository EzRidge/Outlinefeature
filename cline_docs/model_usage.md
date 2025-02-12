# Model Usage and Integration Guide

## Overview
The trained model will provide three key capabilities:
1. Roof Segmentation (from RID dataset)
2. Line Detection (from Roofline dataset)
3. Building Outline (from AIRS dataset)

## Model Outputs
For each input image, the model produces:
```python
predictions = {
    'segments': tensor,  # [B, C, H, W] - Segmentation masks
    'lines': tensor,     # [B, 4, H, W] - Ridge, valley, eave, outline
    'depth': tensor      # [B, H, W] - Pseudo-depth map
}
```

## Integration Steps

### 1. Model Loading
```python
from models import HybridRoofModel

def load_model(checkpoint_path):
    # Load model architecture
    model = HybridRoofModel(num_segment_classes=12)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    return model
```

### 2. Image Processing
```python
def process_image(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (1024, 1024))
    
    # Normalize and convert to tensor
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image
```

### 3. Making Predictions
```python
def predict(model, image):
    with torch.no_grad():
        predictions = model(image)
    return predictions
```

### 4. Post-processing Results

#### Segmentation Masks
```python
def get_segmentation(predictions):
    # Get segment predictions
    segments = predictions['segments']
    
    # Convert to class labels
    masks = torch.argmax(segments, dim=1)
    
    # Convert to numpy
    masks = masks.cpu().numpy()
    return masks
```

#### Line Detection
```python
def get_lines(predictions):
    # Get line predictions
    lines = predictions['lines']
    
    # Split into different line types
    ridge_lines = lines[:, 0]  # Ridge lines
    valley_lines = lines[:, 1]  # Valley lines
    eave_lines = lines[:, 2]   # Eave lines
    outline = lines[:, 3]      # Building outline
    
    return {
        'ridge': ridge_lines.cpu().numpy(),
        'valley': valley_lines.cpu().numpy(),
        'eave': eave_lines.cpu().numpy(),
        'outline': outline.cpu().numpy()
    }
```

#### Depth Estimation
```python
def get_depth(predictions):
    # Get depth predictions
    depth = predictions['depth']
    
    # Convert to numpy
    depth = depth.cpu().numpy()
    return depth
```

### 5. Visualization Tools
```python
def visualize_results(image, predictions):
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0,0].imshow(image)
    axes[0,0].set_title('Original Image')
    
    # Segmentation mask
    masks = get_segmentation(predictions)
    axes[0,1].imshow(masks[0], cmap='tab20')
    axes[0,1].set_title('Segmentation')
    
    # Line detection
    lines = get_lines(predictions)
    line_vis = np.zeros_like(image)
    line_vis[:,:,0] = lines['ridge'][0]    # Red for ridge
    line_vis[:,:,1] = lines['valley'][0]   # Green for valley
    line_vis[:,:,2] = lines['outline'][0]  # Blue for outline
    axes[1,0].imshow(line_vis)
    axes[1,0].set_title('Line Detection')
    
    # Depth map
    depth = get_depth(predictions)
    axes[1,1].imshow(depth[0], cmap='viridis')
    axes[1,1].set_title('Depth Estimation')
    
    plt.tight_layout()
    return fig
```

### 6. Main Project Integration

#### Option 1: REST API
Create a Flask/FastAPI service that:
1. Accepts image uploads
2. Runs model inference
3. Returns JSON with results
4. Provides visualization endpoints

#### Option 2: Python Package
Create a pip-installable package:
```python
from ezridge import RoofAnalyzer

# Initialize analyzer
analyzer = RoofAnalyzer(model_path='path/to/model.pth')

# Analyze single image
results = analyzer.analyze_image('path/to/image.jpg')

# Batch processing
results = analyzer.analyze_directory('path/to/images/')

# Get specific features
segments = results.get_segments()
lines = results.get_lines()
depth = results.get_depth()

# Visualize results
analyzer.visualize(results, save_path='output.png')
```

#### Option 3: Command Line Tool
Create a CLI for batch processing:
```bash
# Analyze single image
ezridge analyze --image path/to/image.jpg --output results/

# Batch process directory
ezridge analyze --input path/to/images/ --output results/

# Generate visualizations
ezridge visualize --results results/ --output visualizations/
```

## Next Steps

1. Local Testing
   - Set up Anaconda environment
   - Test each dataset individually
   - Verify model outputs
   - Create visualization tools

2. Integration Testing
   - Create Python package structure
   - Write integration tests
   - Add example notebooks
   - Document API usage

3. Deployment
   - Package model weights
   - Create Docker container
   - Set up CI/CD pipeline
   - Write deployment docs

4. Documentation
   - API reference
   - Usage examples
   - Best practices
   - Troubleshooting guide