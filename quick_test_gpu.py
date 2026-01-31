"""Quick GPU speedup test - trains 1 fold only"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
from pathlib import Path
import time
from preprocessing import load_and_preprocess_subject
from feature_extraction import extract_all_features, get_feature_names
from main import (
    remap_labels, apply_smote, select_features,
    train_lightgbm, train_xgboost, train_random_forest,
    evaluate
)

print("=" * 70)
print("QUICK GPU SPEEDUP TEST (1 FOLD)")
print("=" * 70)

# Config
DATA_DIR = os.environ.get("EEG_DATA_DIR", "D:/py projects/EEG/Sleep Staging with Forehead EEG")
CACHE_DIR = "cache"
FEATURE_CACHE_DIR = "cache_features"
SFREQ = 500
CONTEXT = 2
SEED = 42
USE_GPU = True

np.random.seed(SEED)

# Get subjects
data_dir = Path(DATA_DIR)
subject_dirs = sorted(data_dir.glob("sub-*"))
unique_subjects = [d.name for d in subject_dirs if d.is_dir()]

print(f"\nFound {len(unique_subjects)} subjects")

# Load features (cached)
subject_features = {}
subject_labels = {}
cache_suffix = f"_v2_ctx{CONTEXT}"

print("\nLoading cached features...")
for subj in unique_subjects[:5]:  # Load only first 5 subjects for quick test
    feat_cache = Path(FEATURE_CACHE_DIR) / f"{subj}{cache_suffix}.npz"
    if feat_cache.exists():
        cached = np.load(feat_cache)
        subject_features[subj] = cached['features']
        subject_labels[subj] = cached['labels']

valid_subjects = list(subject_features.keys())
print(f"Loaded {len(valid_subjects)} subjects from cache")

if len(valid_subjects) < 2:
    print("ERROR: Need at least 2 subjects. Run main.py first to generate cache.")
    exit(1)

# Use first subject as test
test_subject = valid_subjects[0]
train_subjects = valid_subjects[1:]

print(f"\nTest subject: {test_subject}")
print(f"Train subjects: {', '.join(train_subjects)}")

# Build datasets
X_train = np.concatenate([subject_features[s] for s in train_subjects])
y_train = np.concatenate([subject_labels[s] for s in train_subjects])
X_test = subject_features[test_subject]
y_test = subject_labels[test_subject]

print(f"\nTrain: {len(X_train):,} epochs, Test: {len(X_test):,} epochs")

# Apply SMOTE
print("\n[1/4] Applying SMOTE...")
t0 = time.time()
X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, random_state=SEED)
smote_time = time.time() - t0
print(f"      SMOTE: {smote_time:.1f}s")

# Feature selection
print("[2/4] Selecting features...")
t0 = time.time()
feature_names = get_feature_names(n_channels=3, context=CONTEXT)
X_train_selected, X_test_selected, selected_features = select_features(
    X_train_balanced, y_train_balanced, X_test, feature_names,
    method='lgb', importance_threshold='median'
)
selection_time = time.time() - t0
print(f"      Selection: {selection_time:.1f}s")

stage_names = ['Wake', 'REM', 'N1', 'N2', 'N3']

# Test GPU training
print("\n[3/4] Training models with GPU...")
print("-" * 70)

# LightGBM GPU
print("\nLightGBM (GPU)...")
t0 = time.time()
lgb_model = train_lightgbm(X_train_selected, y_train_balanced, use_gpu=True, fast_mode=False)
lgb_time = time.time() - t0
y_pred = lgb_model.predict(X_test_selected)
lgb_metrics = evaluate(y_test, y_pred, stage_names)
print(f"  Time: {lgb_time:.1f}s | Acc: {lgb_metrics['accuracy']:.4f} | F1: {lgb_metrics['f1_macro']:.4f}")

# XGBoost GPU
print("\nXGBoost (GPU)...")
t0 = time.time()
xgb_model = train_xgboost(X_train_selected, y_train_balanced, use_gpu=True, fast_mode=False)
xgb_time = time.time() - t0
y_pred = xgb_model.predict(X_test_selected)
xgb_metrics = evaluate(y_test, y_pred, stage_names)
print(f"  Time: {xgb_time:.1f}s | Acc: {xgb_metrics['accuracy']:.4f} | F1: {xgb_metrics['f1_macro']:.4f}")

# Random Forest CPU (for comparison)
print("\nRandom Forest (CPU)...")
t0 = time.time()
rf_model = train_random_forest(X_train_selected, y_train_balanced, fast_mode=False)
rf_time = time.time() - t0
y_pred = rf_model.predict(X_test_selected)
rf_metrics = evaluate(y_test, y_pred, stage_names)
print(f"  Time: {rf_time:.1f}s | Acc: {rf_metrics['accuracy']:.4f} | F1: {rf_metrics['f1_macro']:.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total time for 1 fold: {smote_time + selection_time + lgb_time + xgb_time + rf_time:.1f}s")
print(f"\nModel training times:")
print(f"  LightGBM (GPU): {lgb_time:.1f}s")
print(f"  XGBoost (GPU):  {xgb_time:.1f}s")
print(f"  RandomForest:   {rf_time:.1f}s")
print(f"\nEstimated time for 19 folds: {(smote_time + selection_time + lgb_time + xgb_time + rf_time) * 19 / 60:.1f} minutes")
print("=" * 70)
