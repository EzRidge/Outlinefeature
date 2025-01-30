# Roof Feature Detection and Measurement

This project implements automatic roof feature detection and measurement from aerial imagery, based on the Roofline-Extraction paper and RID (Roof Information Dataset). It uses deep learning to detect and segment roof outlines, ridge lines, and eave lines, enabling accurate roof measurements.

## Features

- Automatic detection of roof outlines, ridge lines, and eave lines
- Accurate measurement of roof area, perimeter, and pitch
- Support for processing single images or entire directories
- Detailed output including segmentation masks and measurements
- Command-line interface for dataset preparation, training, and inference

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project provides three main commands: prepare, train, and predict.

### Preparing the Dataset

To prepare the RID dataset for training:

```bash
python -m src.main prepare /path/to/rid/dataset /path/to/output --split 0.8 --verify
```

Options:
- `--split`: Train/validation split ratio (default: 0.8)
- `--verify`: Verify the prepared dataset

### Training the Model

To train the model:

```bash
python -m src.main train /path/to/images /path/to/masks --epochs 50 --batch-size 8 --learning-rate 1e-4 --device cuda
```

Options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 8)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--device`: Device to train on ('cuda' or 'cpu', default: 'cuda')

### Running Inference

To run inference on new images:

```bash
python -m src.main predict /path/to/model /path/to/input /path/to/output --device cuda
```

The input path can be either a single image or a directory. Results will be saved to the specified output directory.

## Model Architecture

The model uses a multi-scale CNN architecture based on the Roofline-Extraction paper, with the following key components:

- Encoder: ResNet-based backbone with multi-scale feature extraction
- Feature Fusion: Multi-scale feature fusion module
- Decoder: Upsampling path with skip connections
- Output: Multi-class segmentation (background, roof outline, ridge lines, eave lines)

## Output Format

The prediction output includes:

1. Segmentation masks:
   - `*_roof_mask.png`: Roof outline segmentation
   - `*_ridge_mask.png`: Ridge line segmentation
   - `*_eave_mask.png`: Eave line segmentation

2. Measurements (JSON format):
   - Area (square meters)
   - Perimeter (meters)
   - Pitch (degrees)
   - Number of facets
   - Contour points

## Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- GDAL 3.3+
- Other dependencies listed in requirements.txt

## References

1. Roofline-Extraction paper: [https://www.mdpi.com/2072-4292/11/19/2219](https://www.mdpi.com/2072-4292/11/19/2219)
2. RID Dataset: [https://github.com/TUMFTM/RID](https://github.com/TUMFTM/RID)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
