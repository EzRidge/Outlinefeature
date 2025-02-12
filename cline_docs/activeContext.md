# Active Development Context

## Current Focus: Minimal Test Configuration

### Current State
Running minimal test with:
- CPU instance (ml.m5.xlarge)
- 10 samples from RID dataset only
- Batch size: 2
- Workers: 2
- Image size: 512x512
- 1 epoch

### Recent Changes
1. Added dataset selection:
   - New --datasets parameter
   - Comma-separated list (rid,roofline,airs)
   - Currently using 'rid' only

2. Memory optimization:
   - Reduced batch size to 2
   - Reduced workers to 2
   - Limited to 10 samples
   - Using 512x512 images

3. Documentation:
   - Added scaling_strategy.md
   - Updated progress.md
   - Updated activeContext.md

### Next Actions
1. Monitor current test run:
   - Watch memory usage
   - Check loss values
   - Verify data loading
   - Confirm model saving

2. If successful, proceed to Phase 1 of scaling:
   ```python
   hyperparameters = {
       'epochs-per-dataset': 1,
       'batch-size': 2,
       'learning-rate': 0.001,
       'num-workers': 2,
       'num-classes': 12,
       'max-samples': 20,  # Increase from 10 to 20
       'datasets': 'rid'
   }
   ```

3. Document results in scaling_strategy.md:
   - Memory usage
   - Training time
   - Loss values
   - Any issues encountered

### Key Files
1. Active development:
   - src/sagemaker_train.py
   - src/data.py
   - notebooks/sagemaker_training.ipynb

2. Documentation:
   - cline_docs/scaling_strategy.md
   - cline_docs/progress.md
   - cline_docs/activeContext.md

### Notes
- Keep current minimal configuration until successful run
- Document all metrics for baseline comparison
- Follow scaling strategy phases carefully
- Update documentation after each test
