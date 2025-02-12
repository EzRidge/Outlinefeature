# Active Development Context

## Current Focus: SageMaker Training Pipeline

### Immediate Task
Testing SageMaker training with reduced dataset to verify:
1. Data loading and preprocessing
2. Training loop functionality
3. Loss calculation
4. Memory usage

### Current State
- Using CPU instance (ml.m5.xlarge)
- Reduced settings for testing:
  * Image size: 512x512
  * Batch size: 4
  * Max samples: 20 (to be added)
  * Epochs: 1 per dataset

### Recent Changes
1. Data loading improvements:
   - Fixed file extension handling
   - Added mask value remapping
   - Added max_samples parameter
   - Reduced memory usage

2. Documentation updates:
   - Added sagemaker_training.md
   - Updated progress.md
   - Updated activeContext.md

### Next Actions
1. Add max_samples parameter to training:
   ```python
   hyperparameters = {
       'epochs-per-dataset': 1,
       'batch-size': 4,
       'learning-rate': 0.001,
       'num-workers': 4,
       'num-classes': 12,
       'max-samples': 20  # New parameter
   }
   ```

2. Test training with reduced dataset:
   - Pull latest changes
   - Run training with new parameters
   - Monitor memory usage
   - Check loss values

3. If successful:
   - Document working configuration
   - Plan GPU migration
   - Prepare for full dataset training

### Key Files
1. Active development:
   - src/data.py
   - notebooks/sagemaker_training.ipynb
   - src/sagemaker_train.py

2. Documentation:
   - cline_docs/sagemaker_training.md
   - cline_docs/progress.md
   - cline_docs/activeContext.md

### Notes
- Keep batch size and image size small until testing is complete
- Monitor memory usage in SageMaker logs
- Document any new issues or insights
- Plan for proper mask value mapping in next phase
