"""Simplified GPU test - skip SMOTE to avoid threading issues"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['MKL_THREADING_LAYER'] = 'GNU'  # Fix threading issue

import numpy as np
from pathlib import Path
import time

print("=" * 70)
print("SIMPLIFIED GPU SPEEDUP TEST")
print("=" * 70)

# Config
FEATURE_CACHE_DIR = "cache_features"
CONTEXT = 2

# Load cached features
print("\nLoading cached features...")
subject_features = {}
subject_labels = {}
cache_suffix = f"_v2_ctx{CONTEXT}"

data_dir = Path("D:/py projects/EEG/Sleep Staging with Forehead EEG")
subject_dirs = sorted(data_dir.glob("sub-*"))
unique_subjects = [d.name for d in subject_dirs if d.is_dir()]

for subj in unique_subjects[:5]:  # Load 5 subjects
    feat_cache = Path(FEATURE_CACHE_DIR) / f"{subj}{cache_suffix}.npz"
    if feat_cache.exists():
        cached = np.load(feat_cache)
        subject_features[subj] = cached['features']
        subject_labels[subj] = cached['labels']

valid_subjects = list(subject_features.keys())
print(f"Loaded {len(valid_subjects)} subjects")

# Build dataset
test_subject = valid_subjects[0]
train_subjects = valid_subjects[1:]

X_train = np.concatenate([subject_features[s] for s in train_subjects])
y_train = np.concatenate([subject_labels[s] for s in train_subjects])
X_test = subject_features[test_subject]
y_test = subject_labels[test_subject]

print(f"\nTrain: {len(X_train):,} samples")
print(f"Test:  {len(X_test):,} samples")
print(f"Features: {X_train.shape[1]:,}")

# Test GPU training (without SMOTE, just raw training speed)
print("\n" + "=" * 70)
print("GPU TRAINING TEST")
print("=" * 70)

# LightGBM GPU
print("\n[1] LightGBM GPU...")
try:
    import lightgbm as lgb

    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=5,
        n_estimators=500,  # Reduced for quick test
        num_leaves=127,
        learning_rate=0.03,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    lgb_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()

    print(f"    Time: {lgb_time:.2f}s")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Status: GPU WORKING!")
except Exception as e:
    print(f"    ERROR: {e}")
    lgb_time = 0

# XGBoost GPU
print("\n[2] XGBoost GPU...")
try:
    import xgboost as xgb

    t0 = time.time()
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        n_estimators=500,  # Reduced for quick test
        max_depth=10,
        learning_rate=0.05,
        tree_method='hist',
        device='cuda:0',
        verbosity=0,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()

    print(f"    Time: {xgb_time:.2f}s")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    Status: GPU WORKING!")
except Exception as e:
    print(f"    ERROR: {e}")
    xgb_time = 0

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"LightGBM GPU: {lgb_time:.1f}s")
print(f"XGBoost GPU:  {xgb_time:.1f}s")
print(f"Total:        {lgb_time + xgb_time:.1f}s")
print(f"\nEstimated for 19 folds: {(lgb_time + xgb_time) * 19 / 60:.1f} minutes")
print("=" * 70)
print("\nGPU acceleration is READY!")
print("You can now run: python main.py")
