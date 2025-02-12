# SageMaker Training: Testing and Scaling Strategy

## Current Configuration (Minimal Test)

### Hardware
- Instance: ml.m5.xlarge (CPU)
- Memory: 16 GB
- vCPUs: 4

### Training Parameters
- Image size: 512x512 (reduced from 1024x1024)
- Batch size: 2 (minimal)
- Workers: 2 (minimal)
- Max samples: 10 per dataset
- Datasets: RID only
- Epochs: 1 per dataset

## Scaling Strategy

### Phase 1: Sample Size Scaling
Goal: Verify model can handle more data without memory issues
1. Increase to 20 samples
   - Watch: Memory usage, training time
   - Expect: ~2x training time
   - Success criteria: No OOM errors, stable loss

2. Increase to 50 samples
   - Watch: Memory usage, training time, loss convergence
   - Expect: ~2.5x training time from 20 samples
   - Success criteria: Smooth loss curve

3. Increase to 100 samples
   - Watch: Memory usage, training time, loss convergence
   - Expect: ~2x training time from 50 samples
   - Success criteria: Clear learning progress

### Phase 2: Batch Size Optimization
Goal: Find optimal batch size for current hardware
1. Increase to batch size 4
   - Watch: Memory usage, training speed (samples/second)
   - Expect: Faster training, higher memory usage
   - Success criteria: Training speed improvement

2. Increase to batch size 8
   - Watch: Memory usage, training speed, loss stability
   - Expect: Further speed improvement if memory allows
   - Success criteria: No OOM errors

3. Test batch size 16 (if memory allows)
   - Watch: Memory usage, training speed, loss stability
   - Success criteria: Stable training, good memory margin

### Phase 3: Worker Optimization
Goal: Maximize CPU utilization
1. Increase to 4 workers
   - Watch: CPU usage, data loading speed
   - Expect: Faster data loading
   - Success criteria: Higher CPU utilization

2. Monitor system metrics
   - CPU utilization
   - I/O wait times
   - Memory usage
   - Success criteria: Balanced resource usage

### Phase 4: Dataset Integration
Goal: Test multi-dataset training
1. Add Roofline dataset
   - Watch: Memory usage during transitions
   - Expect: Similar performance to RID
   - Success criteria: Smooth dataset transition

2. Add AIRS dataset
   - Watch: Memory usage, training stability
   - Expect: Similar performance to other datasets
   - Success criteria: Consistent performance across datasets

### Phase 5: GPU Migration (G5 Instance)
Goal: Scale up to full training
1. Initial GPU test
   - Use current best configuration
   - Watch: GPU memory usage, training speed
   - Expect: Significant speed improvement
   - Success criteria: Stable GPU training

2. GPU optimization
   - Increase batch size for GPU
   - Restore 1024x1024 image size
   - Enable mixed precision training
   - Success criteria: Optimal GPU utilization

3. Full dataset training
   - Remove max_samples limit
   - Use all datasets
   - Multiple epochs
   - Success criteria: Complete training pipeline

## Metrics to Watch

### System Metrics
- Memory usage (should stay below 80%)
- CPU/GPU utilization
- Disk I/O
- Network I/O (for data loading)

### Training Metrics
- Loss values (training and validation)
- Training speed (samples/second)
- Time per epoch
- Model accuracy metrics

### Resource Efficiency
- Memory per sample
- Processing time per sample
- GPU memory utilization (when applicable)

## Success Criteria for GPU Migration

Before moving to G5 instance:
1. Stable training on CPU with:
   - At least 100 samples
   - Batch size ≥ 8
   - All datasets working
   - No memory issues

2. Clear understanding of:
   - Memory requirements per sample
   - Optimal batch size
   - Dataset loading patterns
   - Training time expectations

## Documentation Updates

After each phase:
1. Update this document with:
   - Actual results vs expectations
   - Any issues encountered
   - Solutions implemented
   - Recommendations for next phase

2. Update hyperparameters in notebook:
   - Document successful configurations
   - Note any failed attempts
   - Record performance metrics

## Rollback Plan

If issues occur:
1. Return to last working configuration
2. Document the issue
3. Analyze logs and metrics
4. Make smaller incremental changes

## Final Configuration Target

### Hardware
- Instance: ml.g5.xlarge or larger
- GPU: NVIDIA A10G
- Memory: 24GB+

### Training Parameters
- Image size: 1024x1024
- Batch size: 16+
- Workers: 4+
- Samples: Full dataset
- Datasets: All (RID, Roofline, AIRS)
- Epochs: Multiple per dataset