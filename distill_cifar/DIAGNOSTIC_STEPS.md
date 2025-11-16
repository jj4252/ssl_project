# Diagnostic Steps for Knowledge Distillation Issues

This document describes the diagnostic steps implemented to debug poor student model performance.

## Problem Summary

- **Teacher accuracy**: 98.71% (excellent)
- **Projected teacher accuracy**: 98.27% (only 0.45% drop - projection is fine)
- **Untrained student**: 36.94% (reasonable baseline)
- **Trained student**: 11.90% (worse than untrained!)

The student learns diverse features but they're not discriminative for classification.

## Diagnostic Steps Implemented

### Step 1: Verify Representation Consistency ✅

**Purpose**: Ensure training and evaluation use the same feature extraction.

**Implementation**: 
- Automatically runs at the start of training (first batch, first epoch)
- Compares features from `extract_student_features()` (training) vs `forward_features()[:, 0]` (evaluation)
- Prints statistics and warns if they differ

**How to use**: 
- Just run training - it will automatically check and print results
- Look for: `🔍 Step 1 Diagnostic: Training vs Evaluation Representation Consistency`
- If representations differ, this could explain train/eval mismatch

**Expected output**:
```
🔍 Step 1 Diagnostic: Training vs Evaluation Representation Consistency
  Training CLS (extract_student_features): shape=torch.Size([64, 384]), norm=1.000000, std=0.051031
  Eval CLS (forward_features[:, 0]): shape=torch.Size([64, 384]), norm=1.000000, std=0.051031
  ✓ Representations match (diff=0.00000000)
```

---

### Step 2: CLS-Only KD Experiment ✅

**Purpose**: Test if patch loss is interfering with CLS token learning.

**Implementation**:
- Modify `train_config_kd.yaml` to set `distill_loss_weights.patch: 0.0`
- This disables patch loss and only trains on CLS token

**How to use**:
1. Edit `train_config_kd.yaml`:
   ```yaml
   distill_loss_weights:
     cls: 1.0
     patch: 0.0  # Disable patch loss for CLS-only experiment
   ```
2. Train normally: `python distill_trainer.py ...`
3. Evaluate and compare accuracy

**Expected outcome**:
- If CLS-only KD gives better accuracy (even 30-50%), it suggests patch loss was interfering
- If accuracy is still poor, the issue is with CLS loss itself

---

### Step 3: Minimal Augmentation ✅

**Purpose**: Test if heavy augmentations are causing issues.

**Implementation**:
- Added `MinimalTransform` class (resize + flip only, no crop/blur)
- Can be enabled via config

**How to use**:
1. Edit `train_config_kd.yaml`:
   ```yaml
   use_minimal_aug: true  # Enable minimal augmentation
   ```
2. Train normally
3. Compare with default augmentation results

**Expected outcome**:
- If minimal augmentation improves accuracy, heavy augmentations were the issue
- If no change, augmentations are not the problem

---

### Step 4: Supervised Baseline (Not Implemented)

**Purpose**: Verify architecture and evaluation pipeline work correctly.

**Implementation**: 
- Not implemented (would require separate training script)
- Can be done manually by training student with cross-entropy loss on CIFAR10 labels

**How to use** (manual):
1. Create a supervised training script that:
   - Uses same student architecture
   - Trains with cross-entropy loss on CIFAR10 labels
   - Uses same evaluation pipeline
2. If supervised training gets 80-90% accuracy:
   - Architecture is fine ✓
   - Evaluation pipeline is fine ✓
   - The issue is the KD objective ✗

---

## Recommended Testing Order

1. **Step 1** (automatic) - Run training and check representation consistency
2. **Step 2** (CLS-only) - Quick test, likely to reveal issues
3. **Step 4** (supervised) - Strong sanity check if Step 2 doesn't help
4. **Step 3** (minimal aug) - If other steps don't reveal issues

## Current Configuration

See `train_config_kd.yaml` for current settings:
- `distill_loss_weights.cls: 1.0`
- `distill_loss_weights.patch: 0.5` (set to 0.0 for CLS-only)
- `use_minimal_aug: false` (set to true for minimal augmentation)

## Notes

- All diagnostics are non-destructive (can be toggled via config)
- Step 1 runs automatically (no config needed)
- Steps 2 and 3 require config changes
- Step 4 requires manual implementation

