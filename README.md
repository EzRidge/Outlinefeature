# Hybrid Roof Feature Detection

A hybrid approach combining RID's segmentation capabilities with Roofline-Extraction's feature detection for automatic roof analysis from aerial imagery.

## Features

- Automatic detection of roof features:
  - Complete roof outlines
  - Ridge lines
  - Hip lines
  - Valley lines
- Multi-scale feature detection
- Geometric constraint enforcement
- Support for both RID and Roofline-Extraction datasets
- Detailed measurements and visualizations

## Project Structure

```
.
├── data/
│   ├── processed/    # Processed training data
│   ├── raw/         # Raw dataset files
│   └── output/      # Model outputs and visualizations
├── models/          # Saved model weights
├── src/
│   ├── config.py    # Configuration settings
│   ├── models.py    # Model architecture
│   ├── utils.py     # Utility functions
│   ├── prepare_data.py  # Dataset preparation
│   ├── train.py     # Training script
│   ├── quick_test.py    # Testing script
│   └── main.py      # Main CLI interface
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Windows users, install GDAL and other geospatial packages from wheel files:
```bash
# Download appropriate wheel files for your Python version
pip install <path_to_wheel>/GDAL‐<version>‐cp39‐cp39‐win_amd64.whl
pip install <path_to_wheel>/Fiona‐<version>‐cp39‐cp39‐win_amd64.whl
pip install <path_to_wheel>/Shapely‐<version>‐cp39‐cp39‐win_amd64.whl
```

## Dataset Preparation

The system supports both RID and Roofline-Extraction datasets:

1. Prepare Roofline-Extraction dataset:
```bash
python -m src.main prepare --roofline path/to/imdb.mat --output data/processed
```

2. Prepare RID dataset:
```bash
python -m src.main prepare --rid path/to/rid/dataset --output data/processed
```

3. Prepare both datasets:
```bash
python -m src.main prepare --roofline path/to/imdb.mat --rid path/to/rid/dataset --output data/processed
```

## Training

1. Train from scratch:
```bash
python -m src.main train --data_dir data/processed
```

2. Train with pre-trained weights:
```bash
python -m src.main train --data_dir data/processed --weights path/to/weights.pth
```

3. Resume training from checkpoint:
```bash
python -m src.main train --data_dir data/processed --resume path/to/checkpoint.pth
```

4. Train with custom configuration:
```bash
python -m src.main train --data_dir data/processed --config path/to/config.yaml
```

## Testing

1. Test single image:
```bash
python -m src.main test path/to/image.jpg --weights models/best_model.pth --output data/output
```

2. Test directory of images:
```bash
python -m src.main test path/to/images/dir --weights models/best_model.pth --output data/output --batch
```

## Model Architecture

The hybrid model combines:
- Multi-scale CNN from Roofline-Extraction for feature detection
- RID's segmentation approach for accurate boundary detection
- Custom feature heads for ridge, hip, and valley line detection
- Geometric constraint enforcement through angle prediction

## Results

The model outputs:
- Feature masks (outline, ridge, hip, valley)
- Measurements (areas, lengths)
- Visualizations with color-coded features
- Debug information for each detection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RID (Roof Information Dataset) team for their segmentation approach
- Roofline-Extraction authors for their feature detection methodology
- PyTorch and OpenCV communities for their excellent tools
