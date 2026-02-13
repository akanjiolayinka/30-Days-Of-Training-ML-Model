"""
Test script for Day 2: Earthquake & Tsunami Risk Assessment
Validates the complete ML pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DAY 2: EARTHQUAKE & TSUNAMI RISK ASSESSMENT - TEST SCRIPT")
print("="*60)

# 1. Load Data
print("\n[1/7] Loading dataset...")
try:
    et_data = pd.read_csv('data/earthquake_data_tsunami.csv')
    print(f"[OK] Dataset loaded successfully: {et_data.shape}")
    print(f"  Columns: {list(et_data.columns)}")
    print(f"  Missing values: {et_data.isnull().sum().sum()}")
except Exception as e:
    print(f"[ERROR] Loading data: {e}")
    exit(1)

# 2. Feature Engineering
print("\n[2/7] Performing feature engineering...")
try:
    et_data['energy'] = et_data['magnitude'] ** 2
    et_data['felt_vs_measured'] = et_data['mmi'] - et_data['cdi']
    et_data['dmin_km'] = et_data['dmin'] * 111
    et_data['proximity_score'] = 1 / (et_data['dmin_km'] + 1)
    et_data['gap_norm'] = et_data['gap'] / 360

    et_data['depth_category'] = pd.cut(et_data['depth'],
                                        bins=[0, 70, 300, 700],
                                        labels=['shallow', 'intermediate', 'deep'])

    et_data['season'] = et_data['Month'] % 12 // 3 + 1

    et_data = pd.get_dummies(et_data, columns=['depth_category', 'season'], drop_first=True)

    print(f"[OK] Feature engineering complete: {et_data.shape}")
    print(f"  New features: energy, felt_vs_measured, dmin_km, proximity_score, gap_norm")
    print(f"  Encoded features: depth_category, season")
except Exception as e:
    print(f"[ERROR] Feature engineering: {e}")
    exit(1)

# 3. Prepare Data
print("\n[3/7] Preparing features and target...")
try:
    X = et_data.drop(columns=['tsunami'])
    y = et_data['tsunami']

    print(f"[OK] Features (X): {X.shape}")
    print(f"[OK] Target (y): {y.shape}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
except Exception as e:
    print(f"[ERROR] Preparing data: {e}")
    exit(1)

# 4. Train-Test Split
print("\n[4/7] Splitting data...")
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[OK] Training set: {X_train.shape}")
    print(f"[OK] Test set: {X_test.shape}")
except Exception as e:
    print(f"[ERROR] Splitting data: {e}")
    exit(1)

# 5. Train Initial Model
print("\n[5/7] Training initial Random Forest model (200 estimators)...")
try:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"[OK] Model trained successfully")
    print(f"  Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"[ERROR] Training model: {e}")
    exit(1)

# 6. Train Scaled Model
print("\n[6/7] Training scaled Random Forest model (300 estimators)...")
try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_scaled = RandomForestClassifier(n_estimators=300, random_state=42)
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)

    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
    print(f"[OK] Scaled model trained successfully")
    print(f"  Accuracy: {accuracy_scaled:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_scaled))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_5 = importances.sort_values(ascending=False).head(5)
    print("\nTop 5 Feature Importances:")
    for feat, imp in top_5.items():
        print(f"  {feat}: {imp:.4f}")

except Exception as e:
    print(f"[ERROR] Training scaled model: {e}")
    exit(1)

# 7. Test Prediction
print("\n[7/7] Testing prediction on new data...")
try:
    new_data = pd.DataFrame([{
        'magnitude': 7.2,
        'cdi': 5,
        'mmi': 6,
        'sig': 700,
        'nst': 120,
        'dmin': 0.3,
        'gap': 100,
        'depth': 50,
        'latitude': 38.322,
        'longitude': 142.369,
        'Year': 2024,
        'Month': 3,
        'energy': 7.2**2,
        'felt_vs_measured': 6 - 5,
        'dmin_km': 0.3 * 111,
        'proximity_score': 1 / (0.3*111 + 1),
        'gap_norm': 100 / 360,
        'depth_category_intermediate': 0,
        'depth_category_deep': 0,
        'season_2': 0,
        'season_3': 1,
        'season_4': 0
    }])

    new_data = new_data.reindex(columns=X.columns, fill_value=0)
    new_data_scaled = scaler.transform(new_data)

    pred = model_scaled.predict(new_data_scaled)[0]
    pred_proba = model_scaled.predict_proba(new_data_scaled)[0]

    print(f"[OK] Prediction test successful")
    print(f"  Test scenario: Magnitude 7.2, Depth 50km, Near Japan")
    print(f"  Prediction: {'TSUNAMI LIKELY!' if pred == 1 else 'No tsunami expected.'}")
    print(f"  Probabilities: No Tsunami={pred_proba[0]:.2%}, Tsunami={pred_proba[1]:.2%}")

except Exception as e:
    print(f"[ERROR] Prediction test: {e}")
    exit(1)

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("[SUCCESS] All tests passed successfully!")
print(f"[OK] Dataset: {et_data.shape[0]} earthquakes")
print(f"[OK] Features: {X.shape[1]} (including engineered)")
print(f"[OK] Model Accuracy: {accuracy_scaled:.2%}")
print(f"[OK] Pipeline ready for notebook execution")
print("="*60)
