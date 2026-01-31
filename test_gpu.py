"""Quick test to verify GPU support for LightGBM and XGBoost"""
import numpy as np

print("=" * 60)
print("GPU SUPPORT TEST")
print("=" * 60)

# Test 1: XGBoost GPU
print("\n[1] Testing XGBoost GPU...")
try:
    import xgboost as xgb
    print(f"   XGBoost version: {xgb.__version__}")

    # Create small dummy dataset
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # Try GPU training
    params = {
        'tree_method': 'gpu_hist',
        'device': 'cuda:0',
        'objective': 'binary:logistic',
        'n_estimators': 10
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    print("   [OK] XGBoost GPU: WORKING!")

except Exception as e:
    print(f"   [FAIL] XGBoost GPU: {e}")

# Test 2: LightGBM GPU
print("\n[2] Testing LightGBM GPU...")
try:
    import lightgbm as lgb
    print(f"   LightGBM version: {lgb.__version__}")

    # Create small dummy dataset
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # Try GPU training
    params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'objective': 'binary',
        'n_estimators': 10,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    print("   [OK] LightGBM GPU: WORKING!")

except Exception as e:
    print(f"   [FAIL] LightGBM GPU: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
