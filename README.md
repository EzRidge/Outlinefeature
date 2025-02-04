# EZ Ridge - Roof Analysis Model

## Overview
A deep learning model for analyzing roof structures from aerial imagery, combining features from multiple state-of-the-art approaches:
- Line detection (ridge, hip, valley)
- Segment analysis (azimuth angles)
- Superstructure detection (chimneys, dormers, etc.)
- Depth/elevation understanding

## Project Structure
```
.
├── src/
│   ├── analysis/           # Dataset analysis and visualization tools
│   ├── config.py          # Configuration settings
│   ├── data.py           # Data loading and preprocessing
│   ├── models.py         # Neural network architectures
│   ├── prepare_data.py   # Data preparation utilities
│   ├── train.py         # Training scripts
│   └── utils.py         # Helper functions
│
├── Reference Materials/
│   ├── data/             # Datasets (not in repo - download separately)
│   │   ├── RID/          # Roof Information Detection dataset
│   │   └── Roofline-Extraction/  # Roofline detection dataset
│   └── papers/          # Research papers and documentation
│
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Datasets
The model uses two complementary datasets:

### 1. RID (Roof Information Detection)
- 1,880 georeferenced aerial images
- Segment masks with azimuth angles
- Superstructure annotations
- [Download from TUM](https://doi.org/10.14459/2022mp1655470)

### 2. Roofline-Extraction
- 1,670 aerial images (224x224)
- Ridge, hip, and valley line annotations
- Depth maps for 3D understanding
- [Download from source]

## Analysis Tools
Located in `src/analysis/`:
- `visualize_rid_dataset.py`: Visualize RID dataset samples
- `inspect_dataset.py`: Analyze dataset structure
- `download_dataset.py`: Download and verify datasets

## Setup
1. Clone repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download datasets (see above links)
4. Place datasets in appropriate directories under Reference Materials/data/

## Usage
1. Prepare data:
   ```bash
   python src/prepare_data.py
   ```

2. Train model:
   ```bash
   python src/train.py
   ```

3. Make predictions:
   ```bash
   python src/predict.py
   ```

## Dataset Visualizations
- Sample visualizations are available in:
  - `Reference Materials/data/RID/visualizations/`
  - `Reference Materials/data/Roofline-Extraction/visualizations/`
- Each includes original images, masks, and annotations

## License
- Code: MIT License
- RID Dataset: CC-BY-NC
- Roofline-Extraction Dataset: [License]

## References
1. RID Paper: [Link to paper]
2. Roofline-Extraction Paper: [Link to paper]
