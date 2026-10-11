"""
Quick verification and usage script following instructions:
Instructions_for_Collaborator_Phase2_Model.docx
"""
import joblib
import numpy as np

# 1. Load the Phase 2 champion bundle
bundle_path = 'phase3_bundle.pkl'
print(f"Loading {bundle_path}...")
bundle = joblib.load(bundle_path)

# 2. Extract objects
model = bundle['model']
feature_names = bundle['feature_names']
X_test = bundle['X_test']
y_test = bundle['y_test']

print("Bundle successfully loaded!")
print(f"Model: {model}")
print(f"Number of selected features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print(f"Test data shape: {X_test.shape}")
print(f"Target vector shape: {y_test.shape}")

# 3. Quick Verification / Running Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"\nPredictions ready for {len(predictions)} students.")
print(f"Sample predictions (first 5): {predictions[:5]}")
print(f"Sample probabilities (first 5):\n{probabilities[:5]}")
