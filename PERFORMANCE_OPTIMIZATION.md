# Performance Optimization Guide

## 🐌 Why Training is Slow?

### Baseline Performance (CPU Only):
- **Per fold:** ~60-90 seconds
- **Total (19 folds):** ~20-30 minutes
- **With improvements:** ~40-60 minutes (due to more features & SMOTE)

### Performance Bottlenecks:

1. **Feature Extraction (30-40% of time)**
   - Sleep-specific features use Hilbert transform → computationally expensive
   - Temporal context creates 3x more features
   - ~2,700 features before selection

2. **SMOTE (15-20% of time)**
   - Generates 2-3x more samples
   - 18,000 epochs → 40,000+ epochs after balancing

3. **Model Training (40-50% of time)**
   - 4 models per fold (LightGBM, XGBoost, RF, Ensemble)
   - 1,000 trees per model
   - Total: 19 folds × 4 models = 76 models trained

---

## 🚀 GPU Acceleration (MASSIVE Speedup!)

### ✅ **Option 1: LightGBM GPU (Easiest, 5-10x faster)**

#### Requirements:
- NVIDIA GPU with CUDA support
- OpenCL or CUDA installed

#### Installation (Windows with CUDA):
```bash
pip uninstall lightgbm
pip install lightgbm --install-option=--gpu
```

#### Installation (Linux with CUDA):
```bash
pip uninstall lightgbm
cmake .. -DUSE_GPU=1
pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/"
```

#### Enable in code:
Already implemented! Set in [main.py:46](main.py:46):
```python
USE_GPU = True  # Already enabled by default!
```

**Expected speedup:** 5-10x faster (60s → 6-12s per fold)

---

### ✅ **Option 2: XGBoost GPU (10-20x faster)**

#### Requirements:
- NVIDIA GPU with CUDA 11.0+
- XGBoost compiled with GPU support

#### Installation:
```bash
pip uninstall xgboost
pip install xgboost[gpu]
# or for latest version
pip install git+https://github.com/dmlc/xgboost.git@master
```

#### Enable in code:
Already implemented! Set in [main.py:46](main.py:46):
```python
USE_GPU = True  # Already enabled by default!
```

**Expected speedup:** 10-20x faster (60s → 3-6s per fold)

---

### ✅ **Option 3: Random Forest GPU with cuML (RAPIDS)**

#### Requirements:
- NVIDIA GPU with CUDA 11.2+
- Linux or WSL2 (not directly supported on Windows)

#### Installation (Linux/WSL2):
```bash
conda create -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia \
    rapids=23.12 python=3.10 cudatoolkit=11.8
conda activate rapids-23.12
```

#### Modify code to use cuML:
```python
# Replace in main.py
from cuml.ensemble import RandomForestClassifier  # GPU-accelerated!
```

**Expected speedup:** 5-15x faster

**Note:** cuML is complex to set up, only recommended if you need maximum RF speed.

---

## ⚡ Other Optimizations (No GPU Required)

### ✅ **1. Fast Mode (2-3x faster, ~5% accuracy drop)**

Set in [main.py:45](main.py:45):
```python
FAST_MODE = True  # Fewer estimators, faster training
```

**Changes:**
- LightGBM: 1000 → 300 estimators
- XGBoost: 1000 → 300 estimators
- Random Forest: 500 → 200 estimators
- Shallower trees

**Expected impact:**
- Training time: 40 min → 15-20 min
- Accuracy drop: ~3-5%
- Still should achieve >80% accuracy

---

### ✅ **2. Parallel Feature Extraction (Already Enabled!)**

Automatically uses all CPU cores for feature extraction:
```python
# In feature_extraction.py, line 217
features = extract_all_features(epochs, sfreq=500, context=2, n_jobs=-1)
```

**Expected speedup:** 2-4x faster feature extraction (depending on CPU cores)

---

### ✅ **3. Reduce Temporal Context**

In [main.py:44](main.py:44):
```python
CONTEXT = 1  # Change from 2 to 1
```

**Impact:**
- Features: 2,700 → 1,350 (50% reduction)
- Training time: -30-40%
- Accuracy drop: ~2-3%

---

### ✅ **4. Disable Ensemble (Keep Best 2 Models)**

Comment out ensemble in [main.py:234-235](main.py:234-235):
```python
# if len(base_classifiers) >= 2:
#     base_classifiers.append('ensemble')
```

**Impact:**
- Training time: -25%
- Accuracy: Use best individual model (LightGBM or XGBoost)

---

### ✅ **5. Feature Selection with More Aggressive Threshold**

In [main.py:258](main.py:258):
```python
importance_threshold='0.7'  # Change from 'median' (keeps only top 30% features)
```

**Impact:**
- Features kept: 675 → 200
- Training time: -20-30%
- Accuracy drop: ~1-2%

---

### ✅ **6. Cache Everything**

Already implemented! Caching is automatic:
- Raw preprocessed epochs: `cache/`
- Extracted features: `cache_features/`

**On first run:** Slow (feature extraction)
**On subsequent runs:** Fast (loads from cache)

To regenerate cache (if you change features):
```bash
rm -rf cache_features/*
```

---

## 📊 Performance Comparison Table

| Configuration | Time per Fold | Total Time | Expected Acc | Recommended For |
|--------------|---------------|------------|--------------|-----------------|
| **Baseline (CPU, Full)** | 60-90s | 30-40 min | 85-90% | Final results |
| **GPU + Full** | 6-12s | 3-6 min | 85-90% | **Best choice!** |
| **CPU + Fast Mode** | 20-30s | 10-15 min | 80-85% | Quick experiments |
| **GPU + Fast Mode** | 2-4s | 1-2 min | 80-85% | Rapid iteration |
| **CPU + Context=1** | 40-50s | 20-25 min | 83-88% | Good compromise |
| **GPU + Context=1** | 5-8s | 2-4 min | 83-88% | Fast + accurate |

---

## 🎯 Recommended Configurations

### For Quick Testing (Development):
```python
# In main.py
FAST_MODE = True
USE_GPU = True  # if available
CONTEXT = 1
```
**Time:** 1-5 minutes
**Accuracy:** ~80-85%

### For Best Accuracy (Final Run):
```python
# In main.py
FAST_MODE = False
USE_GPU = True  # if available
CONTEXT = 2
```
**Time:** 3-30 minutes (depends on GPU)
**Accuracy:** ~85-90%

### For CPU-Only Systems:
```python
# In main.py
FAST_MODE = True
USE_GPU = False
CONTEXT = 1

# Also comment out XGBoost to save time
# (Remove 'xgboost' from base_classifiers)
```
**Time:** ~10-15 minutes
**Accuracy:** ~80-85%

---

## 🔧 Step-by-Step GPU Setup (Windows)

### 1. Check GPU Compatibility
```bash
nvidia-smi
```
Should show CUDA version (e.g., CUDA 11.8)

### 2. Install CUDA Toolkit (if not installed)
Download from: https://developer.nvidia.com/cuda-downloads

### 3. Install GPU-enabled libraries

**Option A: LightGBM GPU (Easiest)**
```bash
pip uninstall lightgbm
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

**Option B: XGBoost GPU**
```bash
pip uninstall xgboost
pip install xgboost
```
XGBoost should auto-detect GPU if CUDA is installed.

### 4. Verify GPU is Working
```python
import lightgbm as lgb
print(lgb.__version__)

import xgboost as xgb
print(xgb.get_config()['use_gpu'])
```

### 5. Run Training
```bash
python main.py
```

Look for these messages:
```
LightGBM: Using GPU acceleration
XGBoost: Using GPU acceleration
```

---

## 🐛 Troubleshooting

### GPU Not Detected:
1. Check CUDA installation: `nvidia-smi`
2. Reinstall GPU-enabled libraries (see above)
3. Check CUDA version compatibility with library versions

### Out of Memory Errors:
1. Enable FAST_MODE: `FAST_MODE = True`
2. Reduce batch size in SMOTE
3. Reduce number of estimators manually

### Still Slow on GPU:
1. Check GPU utilization: `nvidia-smi -l 1`
2. Ensure data is not too small (GPU overhead)
3. Verify GPU libraries are correctly installed

### CPU Version is Faster Than GPU:
- Small datasets have GPU overhead
- CPU with many cores can outperform GPU for small data
- GPU advantage is for large datasets (>100k samples)

---

## 📈 Expected Performance Gains Summary

**Without any changes (your current setup):**
- ~40-60 minutes total

**With GPU only (no code changes needed):**
- ~3-6 minutes total (**10-20x faster!**)

**With GPU + FAST_MODE:**
- ~1-2 minutes total (**30-50x faster!**)
- ~3-5% accuracy drop

**With parallel feature extraction (already enabled):**
- ~25-30% faster on multi-core CPUs

**Best of all worlds (GPU + optimizations):**
- **1-3 minutes for full 19-fold CV**
- **85-90% accuracy**
- Perfect for rapid iteration!

---

## 💡 Pro Tips

1. **First run will be slow** (feature extraction + caching)
   - Subsequent runs load from cache → much faster

2. **Use FAST_MODE during development**
   - Switch to full mode only for final results

3. **GPU is worth it for this project**
   - 10-20x speedup is huge for experimentation
   - Allows trying many hyperparameter combinations

4. **Monitor GPU usage**
   - Run `nvidia-smi -l 1` in another terminal
   - GPU utilization should be 80-100%

5. **Batch process multiple experiments**
   - Modify hyperparameters
   - Run overnight with different configurations

---

**Questions?** Check:
- LightGBM GPU docs: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
- XGBoost GPU docs: https://xgboost.readthedocs.io/en/latest/gpu/index.html
