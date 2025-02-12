# SageMaker Training Setup and Progress

## Current Status
- Working on getting the model training working on SageMaker
- Using CPU instance (ml.m5.xlarge) for initial testing
- Will move to GPU instance (G5) once everything is working

## Recent Changes
1. Fixed file extension handling:
   - Images are .tif in images_roof_centered_geotiff/
   - Masks are .png in masks_segments_reviewed/
   - Split files contain .png extensions that need to be handled

2. Fixed mask value mapping:
   - RID dataset has values 0-17 for azimuth classes
   - Currently converting to binary (0: background, 1: roof)
   - Will add more sophisticated mapping later

3. Reduced memory usage:
   - Image size reduced from 1024x1024 to 512x512
   - Batch size reduced from 16 to 4
   - Added max_samples parameter for testing

## Next Steps
1. Test with small sample size:
   ```python
   hyperparameters = {
       'epochs-per-dataset': 1,
       'batch-size': 4,
       'learning-rate': 0.001,
       'num-workers': 4,
       'num-classes': 12,
       'max-samples': 20  # Use small number for testing
   }
   ```

2. Once working:
   - Scale up to full dataset
   - Move to G5 GPU instance
   - Restore original image size (1024x1024)
   - Increase batch size (16 or higher)
   - Add proper mask value mapping

## Issues Fixed
1. File extension mismatch between split files and actual files
2. Target out of bounds error (mask values 0-17 vs expected 0-11)
3. Memory issues with large images and batch size
4. Non-tensor values in targets dictionary

## Current Configuration
```python
# Dataset settings
- Image size: 512x512 (reduced for testing)
- Using images_roof_centered_geotiff for full dataset
- Binary mask conversion for RID dataset

# Training settings
- Instance: ml.m5.xlarge (CPU)
- Batch size: 4
- Workers: 4
- Learning rate: 0.001
- Epochs per dataset: 1 (for testing)
```

## Files Modified
1. src/data.py:
   - Added max_samples parameter
   - Fixed file extension handling
   - Added mask value remapping
   - Reduced image size and batch size

2. notebooks/sagemaker_training.ipynb:
   - Updated hyperparameters
   - Using CPU instance for testing
   - Added max_samples parameter

## Repository Structure
```
src/
  ├── data.py          # Dataset loading and preprocessing
  ├── models.py        # Model architecture and loss functions
  └── sagemaker_train.py # SageMaker training script

notebooks/
  └── sagemaker_training.ipynb # Main training notebook
```

## S3 Data Structure
```
s3://{bucket}/roof-data/
  ├── RID/
  │   ├── images_roof_centered_geotiff/    # .tif files
  │   ├── masks_segments_reviewed/         # .png files
  │   └── filenames_train_val_test_split/  # Split files
  ├── Roofline/
  └── AIRS/