"""
Traditional ML training script for EEG Sleep Staging Classification
LightGBM + XGBoost + Random Forest with LOSO cross-validation
Includes: SMOTE, Feature Selection, Ensemble Stacking
"""
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MKL_THREADING_LAYER'] = 'GNU'  # Fix threading compatibility issue

# Suppress compilation warnings from GPU libraries
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
import json
import time
from datetime import datetime
from tqdm import tqdm

# SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imbalanced-learn not installed. SMOTE will not be available.")

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: lightgbm not installed.")

# XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed.")

from preprocessing import load_and_preprocess_subject
from feature_extraction import extract_all_features, get_feature_names

N_CLASSES = 5
CONTEXT = 2  # temporal context: 2 epochs before + 2 after

# Performance configuration
FAST_MODE = False  # Set to True for faster training (lower accuracy)
USE_GPU = True      # GPU acceleration enabled for RTX 5060 (CUDA 12.8)


def remap_labels(labels):
    """Remap labels to 0-indexed (0..4)."""
    min_label = int(labels.min())
    max_label = int(labels.max())

    if min_label == 1 and max_label == 5:
        return labels - 1
    elif min_label == 0 and max_label == 4:
        return labels
    else:
        raise ValueError(
            f"Unexpected label range [{min_label}, {max_label}]. "
            f"Expected [1,5] or [0,4]."
        )


def compute_class_weights(labels, n_classes=5):
    """Compute balanced class weights."""
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    total = len(labels)
    weights = total / (n_classes * counts + 1e-12)
    return {i: w for i, w in enumerate(weights)}


def train_lightgbm(X_train, y_train, use_weights=False, class_weights=None, use_gpu=True, fast_mode=False):
    """Train LightGBM classifier with optimized hyperparameters.

    Args:
        X_train: training features
        y_train: training labels
        use_weights: whether to use class weights (default False, use SMOTE instead)
        class_weights: class weight dictionary (optional)
        use_gpu: whether to use GPU acceleration (default True)
        fast_mode: whether to use faster but less accurate settings (default False)
    """
    sample_weights = None
    if use_weights and class_weights is not None:
        sample_weights = np.array([class_weights[int(y)] for y in y_train])

    # Optimized hyperparameters for better performance
    if fast_mode:
        # Fast mode: fewer estimators, smaller trees
        params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'n_estimators': 300,      # Reduced from 1000
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'max_depth': 10,          # Reduced from 15
            'min_split_gain': 0.01,
            'verbose': -1,
            'random_state': 42,
        }
    else:
        # Full mode: best accuracy
        params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.03,
            'n_estimators': 1000,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_depth': 15,
            'min_split_gain': 0.01,
            'verbose': -1,
            'random_state': 42,
        }

    # GPU acceleration (MUCH faster!)
    if use_gpu:
        try:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
            print("    LightGBM: Using GPU acceleration")
        except Exception as e:
            print(f"    LightGBM: GPU not available, using CPU ({e})")
            params['n_jobs'] = 28
    else:
        params['n_jobs'] = 28

    model = lgb.LGBMClassifier(**params)

    # Suppress GPU compilation warnings
    if use_gpu:
        with suppress_stdout_stderr():
            model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    return model


def train_random_forest(X_train, y_train, use_weights=False, fast_mode=False):
    """Train Random Forest classifier with optimized hyperparameters.

    Args:
        X_train: training features
        y_train: training labels
        use_weights: whether to use class weights (default False, use SMOTE instead)
        fast_mode: whether to use faster but less accurate settings (default False)
    """
    if fast_mode:
        model = RandomForestClassifier(
            n_estimators=200,            # Reduced from 500
            max_depth=15,                # Reduced from 20
            min_samples_split=10,        # Increased for faster training
            min_samples_leaf=4,          # Increased for faster training
            max_features='sqrt',
            class_weight='balanced' if use_weights else None,
            n_jobs=8,
            random_state=42,
            bootstrap=True,
            oob_score=False,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced' if use_weights else None,
            n_jobs=8,
            random_state=42,
            bootstrap=True,
            oob_score=False,
        )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, use_weights=False, class_weights=None, use_gpu=True, fast_mode=False):
    """Train XGBoost classifier with optimized hyperparameters.

    Args:
        X_train: training features
        y_train: training labels
        use_weights: whether to use class weights (default False, use SMOTE instead)
        class_weights: class weight dictionary (optional)
        use_gpu: whether to use GPU acceleration (default True)
        fast_mode: whether to use faster but less accurate settings (default False)
    """
    sample_weights = None
    if use_weights and class_weights is not None:
        sample_weights = np.array([class_weights[int(y)] for y in y_train])

    # Optimized hyperparameters
    if fast_mode:
        params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'n_estimators': 300,      # Reduced from 1000
            'max_depth': 8,           # Reduced from 10
            'learning_rate': 0.1,     # Increased for faster convergence
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_weight': 5,
            'gamma': 0.1,
            'random_state': 42,
            'verbosity': 0,
        }
    else:
        params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_weight': 5,
            'gamma': 0.1,
            'random_state': 42,
            'verbosity': 0,
        }

    # GPU acceleration (10-20x faster!)
    if use_gpu:
        try:
            # XGBoost 2.x+ recommends: tree_method='hist' + device='cuda'
            params['tree_method'] = 'hist'
            params['device'] = 'cuda:0'
            print("    XGBoost: Using GPU acceleration")
        except Exception as e:
            print(f"    XGBoost: GPU not available, using CPU ({e})")
            params['tree_method'] = 'hist'
            params['n_jobs'] = 8
    else:
        params['tree_method'] = 'hist'
        params['n_jobs'] = 8

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    return model


def evaluate(y_true, y_pred, stage_names):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    return {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'f1_per_class': {name: float(f1) for name, f1 in zip(stage_names, f1_per_class)},
    }


def learn_transition_matrix(labels, n_classes=5):
    """Learn transition probability matrix from training labels."""
    trans = np.zeros((n_classes, n_classes))
    for i in range(len(labels) - 1):
        trans[int(labels[i]), int(labels[i + 1])] += 1
    # Normalize rows (add small smoothing to avoid zeros)
    trans += 1e-3
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance classes.

    Args:
        X_train: training features
        y_train: training labels
        random_state: random seed

    Returns:
        Resampled X_train and y_train
    """
    if not HAS_SMOTE:
        print("  Warning: SMOTE not available, using original data")
        return X_train, y_train

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Before SMOTE: {dict(zip(unique, counts))}")

    # Apply SMOTE with k_neighbors based on smallest class
    min_samples = counts.min()
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    smote = SMOTE(
        sampling_strategy='auto',  # Resample all minority classes
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"  After SMOTE:  {dict(zip(unique, counts))}")

    return X_resampled, y_resampled


def select_features(X_train, y_train, X_test, feature_names, method='lgb', importance_threshold='median'):
    """Feature selection using model importance.

    Args:
        X_train: training features
        y_train: training labels
        X_test: test features
        feature_names: list of feature names
        method: 'lgb' or 'rf' for feature importance
        importance_threshold: threshold for feature selection

    Returns:
        X_train_selected, X_test_selected, selected_feature_names
    """
    # Train a quick model for feature importance
    if method == 'lgb' and HAS_LGBM:
        selector_model = lgb.LGBMClassifier(
            n_estimators=100, num_leaves=31, random_state=42, verbose=-1, n_jobs=8
        )
    else:
        selector_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=8
        )

    selector_model.fit(X_train, y_train)

    # Select features
    selector = SelectFromModel(selector_model, threshold=importance_threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    mask = selector.get_support()
    selected_names = [name for name, selected in zip(feature_names, mask) if selected]

    n_selected = X_train_selected.shape[1]
    n_total = X_train.shape[1]
    print(f"  Feature selection: {n_selected}/{n_total} features kept ({100*n_selected/n_total:.1f}%)")

    return X_train_selected, X_test_selected, selected_names


def hmm_smooth(pred_proba, transition_matrix):
    """Viterbi-like forward smoothing using log probabilities.

    Args:
        pred_proba: (n_epochs, n_classes) classifier probabilities
        transition_matrix: (n_classes, n_classes) transition probabilities

    Returns:
        Smoothed predictions (n_epochs,)
    """
    n_epochs, n_classes = pred_proba.shape
    log_trans = np.log(transition_matrix + 1e-12)
    log_emit = np.log(pred_proba + 1e-12)

    # Viterbi
    viterbi = np.zeros((n_epochs, n_classes))
    backptr = np.zeros((n_epochs, n_classes), dtype=int)

    # Init
    prior = np.ones(n_classes) / n_classes
    viterbi[0] = np.log(prior) + log_emit[0]

    for t in range(1, n_epochs):
        for s in range(n_classes):
            scores = viterbi[t - 1] + log_trans[:, s]
            backptr[t, s] = np.argmax(scores)
            viterbi[t, s] = scores[backptr[t, s]] + log_emit[t, s]

    # Backtrace
    path = np.zeros(n_epochs, dtype=int)
    path[-1] = np.argmax(viterbi[-1])
    for t in range(n_epochs - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path


def main():
    """Main training pipeline"""

    # Configuration
    DATA_DIR = os.environ.get("EEG_DATA_DIR", "D:/py projects/EEG/Sleep Staging with Forehead EEG")
    CACHE_DIR = "cache"
    FEATURE_CACHE_DIR = "cache_features"
    RESULTS_DIR = "results"
    SEED = 42
    SFREQ = 500

    np.random.seed(SEED)

    # Create directories
    for d in [RESULTS_DIR, CACHE_DIR, FEATURE_CACHE_DIR]:
        Path(d).mkdir(exist_ok=True, parents=True)

    # Get list of subjects
    print("=" * 80)
    print("SCANNING DATASET")
    print("=" * 80)
    data_dir = Path(DATA_DIR)
    subject_dirs = sorted(data_dir.glob("sub-*"))
    unique_subjects = [d.name for d in subject_dirs if d.is_dir()]

    print(f"Found {len(unique_subjects)} subjects")
    print(f"Subjects: {', '.join(unique_subjects)}")

    stage_names = ['Wake', 'REM', 'N1', 'N2', 'N3']
    feature_names = get_feature_names(n_channels=3, context=CONTEXT)
    print(f"Features per epoch: {len(feature_names)} (context={CONTEXT})")

    # Precompute features for all subjects (cached)
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION")
    print("=" * 80)

    subject_features = {}
    subject_labels = {}

    cache_suffix = f"_v2_ctx{CONTEXT}"
    for subj in tqdm(unique_subjects, desc="Extracting features"):
        feat_cache = Path(FEATURE_CACHE_DIR) / f"{subj}{cache_suffix}.npz"
        if feat_cache.exists():
            cached = np.load(feat_cache)
            subject_features[subj] = cached['features']
            subject_labels[subj] = cached['labels']
            continue

        try:
            epochs, labels, _ = load_and_preprocess_subject(subj, DATA_DIR, cache_dir=CACHE_DIR)
            labels = remap_labels(labels)
            features = extract_all_features(epochs, sfreq=SFREQ, context=CONTEXT, n_jobs=8)

            np.savez(feat_cache, features=features, labels=labels)
            subject_features[subj] = features
            subject_labels[subj] = labels
        except Exception as e:
            print(f"  {subj}: ERROR - {e}")

    valid_subjects = [s for s in unique_subjects if s in subject_features]
    print(f"\nSuccessfully loaded {len(valid_subjects)} subjects")

    # LOSO CV
    print("\n" + "=" * 80)
    print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
    print("=" * 80)

    # Configure available classifiers
    base_classifiers = []
    if HAS_LGBM:
        base_classifiers.append('lightgbm')
    if HAS_XGB:
        base_classifiers.append('xgboost')
    base_classifiers.append('random_forest')

    # Add ensemble if multiple classifiers available
    if len(base_classifiers) >= 2:
        base_classifiers.append('ensemble')

    # Each base classifier also gets an HMM-smoothed variant
    classifiers = []
    for c in base_classifiers:
        classifiers.append(c)
        classifiers.append(f"{c}+hmm")

    print(f"Classifiers: {', '.join(classifiers)}")
    print(f"SMOTE: {'Enabled' if HAS_SMOTE else 'Disabled'}")
    print(f"Feature Selection: Enabled")

    all_results = {clf: [] for clf in classifiers}

    total_start_time = time.time()
    for fold, test_subject in enumerate(tqdm(valid_subjects, desc="LOSO Folds")):
        fold_start_time = time.time()

        print(f"\n{'=' * 80}")
        print(f"FOLD {fold + 1}/{len(valid_subjects)} - Test: {test_subject}")
        print(f"{'=' * 80}")

        # Build train/test sets
        train_subjects = [s for s in valid_subjects if s != test_subject]
        X_train = np.concatenate([subject_features[s] for s in train_subjects])
        y_train = np.concatenate([subject_labels[s] for s in train_subjects])
        X_test = subject_features[test_subject]
        y_test = subject_labels[test_subject]

        print(f"Train: {len(X_train):,} epochs, Test: {len(X_test):,} epochs")

        # Apply SMOTE for class balancing
        print(f"\n  [1/4] Applying SMOTE...")
        smote_start = time.time()
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, random_state=SEED)
        smote_time = time.time() - smote_start
        print(f"        ✓ SMOTE completed in {smote_time:.1f}s")

        # Feature selection (use balanced data for better selection)
        print(f"  [2/4] Selecting features...")
        selection_start = time.time()
        X_train_selected, X_test_selected, selected_features = select_features(
            X_train_balanced, y_train_balanced, X_test, feature_names,
            method='lgb' if HAS_LGBM else 'rf',
            importance_threshold='median'
        )
        selection_time = time.time() - selection_start
        print(f"        ✓ Feature selection completed in {selection_time:.1f}s")

        class_weights = compute_class_weights(y_train_balanced)

        # Learn transition matrix from training labels (use balanced data)
        trans_matrix = learn_transition_matrix(y_train_balanced, n_classes=N_CLASSES)

        # Store trained models for ensemble
        trained_models = {}

        print(f"  [3/4] Training base models...")
        for clf_name in base_classifiers:
            if clf_name == 'ensemble':
                continue  # Handle ensemble separately

            print(f"\n    Training {clf_name}...", end=' ')
            model_start = time.time()

            # Train model (use SMOTE-balanced data, no class weights)
            if clf_name == 'lightgbm':
                model = train_lightgbm(X_train_selected, y_train_balanced, use_weights=False,
                                      use_gpu=USE_GPU, fast_mode=FAST_MODE)
            elif clf_name == 'xgboost':
                model = train_xgboost(X_train_selected, y_train_balanced, use_weights=False,
                                     use_gpu=USE_GPU, fast_mode=FAST_MODE)
            else:  # random_forest
                model = train_random_forest(X_train_selected, y_train_balanced, use_weights=False,
                                           fast_mode=FAST_MODE)

            model_time = time.time() - model_start
            trained_models[clf_name] = model

            # Raw predictions
            y_pred = model.predict(X_test_selected)
            metrics = evaluate(y_test, y_pred, stage_names)
            print(f"({model_time:.1f}s)")
            print(f"      {clf_name}: Acc={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}  Kappa={metrics['kappa']:.4f}")

            all_results[clf_name].append({
                'fold': fold + 1,
                'test_subject': test_subject,
                **metrics,
            })

            # HMM-smoothed predictions
            pred_proba = model.predict_proba(X_test_selected)
            y_pred_hmm = hmm_smooth(pred_proba, trans_matrix)
            metrics_hmm = evaluate(y_test, y_pred_hmm, stage_names)
            hmm_name = f"{clf_name}+hmm"
            print(f"      {hmm_name}: Acc={metrics_hmm['accuracy']:.4f}  F1={metrics_hmm['f1_macro']:.4f}  Kappa={metrics_hmm['kappa']:.4f}")

            all_results[hmm_name].append({
                'fold': fold + 1,
                'test_subject': test_subject,
                **metrics_hmm,
            })

        # Train ensemble if multiple models available
        if 'ensemble' in base_classifiers and len(trained_models) >= 2:
            print(f"\n  [4/4] Training ensemble...", end=' ')
            ensemble_start = time.time()

            # Build voting ensemble (soft voting using predict_proba)
            estimators = [(name, model) for name, model in trained_models.items()]

            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',  # Use probability predictions
                n_jobs=8
            )

            ensemble.fit(X_train_selected, y_train_balanced)
            ensemble_time = time.time() - ensemble_start
            print(f"({ensemble_time:.1f}s)")

            # Ensemble predictions
            y_pred_ensemble = ensemble.predict(X_test_selected)
            metrics_ensemble = evaluate(y_test, y_pred_ensemble, stage_names)
            print(f"    ensemble: Acc={metrics_ensemble['accuracy']:.4f}  F1={metrics_ensemble['f1_macro']:.4f}  Kappa={metrics_ensemble['kappa']:.4f}")

            all_results['ensemble'].append({
                'fold': fold + 1,
                'test_subject': test_subject,
                **metrics_ensemble,
            })

            # Ensemble with HMM smoothing
            pred_proba_ensemble = ensemble.predict_proba(X_test_selected)
            y_pred_ensemble_hmm = hmm_smooth(pred_proba_ensemble, trans_matrix)
            metrics_ensemble_hmm = evaluate(y_test, y_pred_ensemble_hmm, stage_names)
            print(f"    ensemble+hmm: Acc={metrics_ensemble_hmm['accuracy']:.4f}  F1={metrics_ensemble_hmm['f1_macro']:.4f}  Kappa={metrics_ensemble_hmm['kappa']:.4f}")

            all_results['ensemble+hmm'].append({
                'fold': fold + 1,
                'test_subject': test_subject,
                **metrics_ensemble_hmm,
            })

        # Print fold summary
        fold_time = time.time() - fold_start_time
        print(f"\n  {'─' * 76}")
        print(f"  Fold {fold + 1} completed in {fold_time:.1f}s ({fold_time/60:.1f} min)")
        print(f"  {'─' * 76}")

    # Print total training time
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average time per fold: {total_time/len(valid_subjects):.1f}s")
    print(f"{'=' * 80}")

    # Save results per classifier
    results_dir = Path(RESULTS_DIR)
    for clf_name in classifiers:
        results = all_results[clf_name]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_macro'] for r in results]
        kappas = [r['kappa'] for r in results]

        print(f"\n{'=' * 80}")
        print(f"RESULTS: {clf_name}")
        print(f"{'=' * 80}")
        print(f"Accuracy:  {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"Macro F1:  {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
        print(f"Kappa:     {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")

        print(f"\nAverage per-class F1:")
        for stage in stage_names:
            f1s = [r['f1_per_class'][stage] for r in results]
            print(f"  {stage:6s}: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

        overall = {
            'timestamp': datetime.now().isoformat(),
            'classifier': clf_name,
            'n_folds': len(results),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_f1_macro': float(np.mean(f1_scores)),
            'std_f1_macro': float(np.std(f1_scores)),
            'mean_kappa': float(np.mean(kappas)),
            'std_kappa': float(np.std(kappas)),
            'per_class_f1': {
                stage: {
                    'mean': float(np.mean([r['f1_per_class'][stage] for r in results])),
                    'std': float(np.std([r['f1_per_class'][stage] for r in results]))
                }
                for stage in stage_names
            },
            'fold_results': results,
        }

        with open(results_dir / f'results_{clf_name}.json', 'w') as f:
            json.dump(overall, f, indent=2)

    print(f"\nResults saved to {results_dir.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
