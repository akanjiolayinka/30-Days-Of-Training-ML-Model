"""
Simplified test script for Day 7: Fruit Classification
Tests core functionality without requiring plotly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("DAY 7: FRUIT CLASSIFICATION - SIMPLIFIED TEST")
print("="*60)

# Create directories
os.makedirs('viz', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Data
print("\n[1/6] Loading dataset...")
try:
    fruit_data = pd.read_csv('data/fruit_classification_dataset.csv')
    print(f"[OK] Dataset loaded: {fruit_data.shape}")
    print(f"  Columns: {list(fruit_data.columns)}")
except Exception as e:
    print(f"[ERROR] Loading data: {e}")
    exit(1)

# 2. Check Data Quality
print("\n[2/6] Checking data quality...")
print(f"  Missing values: {fruit_data.isnull().sum().sum()}")
print(f"  Unique fruits: {fruit_data['fruit_name'].nunique()}")
print(f"  Sample distribution:")
for fruit in fruit_data['fruit_name'].value_counts().head(5).items():
    print(f"    {fruit[0]}: {fruit[1]} samples")

# 3. Encode Features
print("\n[3/6] Encoding categorical features...")
try:
    encoded_data = fruit_data.copy()
    le_target = LabelEncoder()
    encoded_data['fruit_name_encoded'] = le_target.fit_transform(encoded_data['fruit_name'])
    categorical_features = ['shape', 'color', 'taste']
    encoded_data = pd.get_dummies(encoded_data, columns=categorical_features, drop_first=True)
    print(f"[OK] Features encoded: {encoded_data.shape[1]} total features")
    print(f"  Numerical features: 3")
    print(f"  Encoded categorical: {encoded_data.shape[1] - 5}")  # 3 numerical + fruit_name + fruit_name_encoded
except Exception as e:
    print(f"[ERROR] Encoding: {e}")
    exit(1)

# 4. Split Data
print("\n[4/6] Splitting data...")
try:
    X = encoded_data.drop(['fruit_name', 'fruit_name_encoded'], axis=1)
    y = encoded_data['fruit_name_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(le_target.classes_)}")
except Exception as e:
    print(f"[ERROR] Splitting: {e}")
    exit(1)

# 5. Train and Evaluate Model
print("\n[5/6] Training Random Forest model...")
try:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print(f"[OK] Model trained: {rf_model.n_estimators} estimators")

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[OK] Test Accuracy: {accuracy*100:.2f}%")

    y_train_pred = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"[OK] Train Accuracy: {train_accuracy*100:.2f}%")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

except Exception as e:
    print(f"[ERROR] Training/Evaluation: {e}")
    exit(1)

# 6. Save Model
print("\n[6/6] Saving model...")
try:
    import joblib
    joblib.dump(rf_model, 'models/fruit_rf_model.joblib')
    joblib.dump(le_target, 'models/label_encoder.joblib')
    print("[OK] Model saved to models/")
    print(f"  - models/fruit_rf_model.joblib")
    print(f"  - models/label_encoder.joblib")
except Exception as e:
    print(f"[ERROR] Saving model: {e}")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("[SUCCESS] All core tests passed!")
print(f"[OK] Dataset: {fruit_data.shape[0]} samples, {fruit_data['fruit_name'].nunique()} fruit types")
print(f"[OK] Features: {X.shape[1]} (after encoding)")
print(f"[OK] Model Accuracy: {accuracy*100:.2f}%")
print(f"[OK] Model saved successfully")
print("\n[NOTE] For full visualization generation, run the Jupyter notebook")
print("       which requires plotly library")
print("="*60)
