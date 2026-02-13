"""
Day 29: Car Price Prediction - Execution Script

This script runs the complete car price prediction analysis cell by cell,
generating all visualizations and trained models.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('viz', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*80)
print("DAY 29: CAR PRICE PREDICTION - EXECUTION STARTED")
print("="*80)

# Cell 1: Import Libraries
print("\n[1/14] Importing libraries...")
import pandas as pd
import numpy as np
import joblib

# Try Plotly first, fallback to matplotlib
try:
    import plotly.express as px
    import plotly.graph_objects as go
    USE_PLOTLY = True
    print("[OK] Using Plotly for visualizations")
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    USE_PLOTLY = False
    print("[OK] Using Matplotlib for visualizations (Plotly not available)")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost not available, will skip XGBoost model")

print("[OK] Libraries imported successfully!")

# Cell 2: Load Dataset
print("\n[2/14] Loading dataset...")
cars = pd.read_csv("data/car_price_prediction_.csv")
print(f"[OK] Dataset loaded: {cars.shape}")
print(f"[INFO] Features: {list(cars.columns)}")
print(f"[INFO] Missing values: {cars.isnull().sum().sum()}")

# Cell 3: Feature Engineering
print("\n[3/14] Engineering features...")
cars['Car_Age'] = 2025 - cars['Year']
cars["Mileage_per_Year"] = cars["Mileage"] / (cars["Car_Age"] + 1)

luxury_brands = ["Tesla", "BMW", "Audi", "Mercedes"]
cars["Is_Luxury"] = cars["Brand"].isin(luxury_brands).astype(int)

cars["Fuel_Group"] = cars["Fuel Type"].replace({
    "Petrol": "Traditional",
    "Diesel": "Traditional",
    "Hybrid": "Eco",
    "Electric": "Eco"
})

print(f"[OK] Feature engineering complete!")
print(f"[INFO] New features: Car_Age, Mileage_per_Year, Is_Luxury, Fuel_Group")
print(f"[INFO] Total features: {cars.shape[1]}")

# Cell 4: Visualization 1 - Brand Distribution
print("\n[4/14] Creating brand distribution visualization...")
if USE_PLOTLY:
    fig = px.histogram(cars, x='Brand',
                       title='Distribution of Car Brands',
                       labels={'Brand': 'Car Brand', 'count': 'Number of Cars'},
                       template='plotly_dark')
    fig.update_layout(bargap=0.2)
    fig.write_html('viz/brand_distribution.html')
    print(f"[OK] Saved: viz/brand_distribution.html")
else:
    plt.figure(figsize=(12, 6))
    cars['Brand'].value_counts().plot(kind='bar', color='steelblue')
    plt.title('Distribution of Car Brands', fontsize=16, fontweight='bold')
    plt.xlabel('Car Brand', fontsize=12)
    plt.ylabel('Number of Cars', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('viz/brand_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: viz/brand_distribution.png")

# Cell 5: Visualization 2 - Average Price by Brand
print("\n[5/14] Creating average price by brand visualization...")
avg_price_by_brand = cars.groupby('Brand')['Price'].mean().reset_index()

if USE_PLOTLY:
    fig = px.bar(avg_price_by_brand,
                 x='Brand', y='Price',
                 title='Average Price by Car Brand',
                 labels={'Brand': 'Car Brand', 'Price': 'Average Price'},
                 template='plotly_dark')
    fig.write_html('viz/avg_price_by_brand.html')
    print(f"[OK] Saved: viz/avg_price_by_brand.html")
else:
    plt.figure(figsize=(12, 6))
    plt.bar(avg_price_by_brand['Brand'], avg_price_by_brand['Price'], color='coral')
    plt.title('Average Price by Car Brand', fontsize=16, fontweight='bold')
    plt.xlabel('Car Brand', fontsize=12)
    plt.ylabel('Average Price', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('viz/avg_price_by_brand.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: viz/avg_price_by_brand.png")

# Cell 6: Prepare data for modeling
print("\n[6/14] Preparing data for modeling...")
X = cars.drop(["Price", "Car ID", "Model", "Engine Size", "Condition"], axis=1)
y = cars["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

print(f"[OK] Training set: {X_train.shape}")
print(f"[OK] Testing set: {X_test.shape}")

# Cell 7: Preprocessing Pipeline
print("\n[7/14] Creating preprocessing pipeline...")
categorical_features = ['Brand', 'Fuel Type', 'Transmission', 'Fuel_Group']
numerical_features = ['Year', 'Mileage', 'Car_Age', 'Mileage_per_Year', 'Is_Luxury']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"[OK] Data preprocessing complete!")
print(f"[OK] Transformed features: {X_train_processed.shape[1]}")

# Cell 8: Train Lasso Regression
print("\n[8/14] Training Lasso Regression...")
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_processed, y_train)
y_pred_lasso = lasso.predict(X_test_processed)

lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

print(f"[OK] Lasso Regression trained!")
print(f"     MAE: {lasso_mae:.2f}")
print(f"     MSE: {lasso_mse:.2f}")
print(f"     R²: {lasso_r2:.4f}")

# Cell 9: Train Random Forest
print("\n[9/14] Training Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=100, random_state=69, n_jobs=-1)
rf.fit(X_train_processed, y_train)
y_pred_rf = rf.predict(X_test_processed)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"[OK] Random Forest trained!")
print(f"     MAE: {rf_mae:.2f}")
print(f"     MSE: {rf_mse:.2f}")
print(f"     R²: {rf_r2:.4f}")

# Cell 10: Train Gradient Boosting
print("\n[10/14] Training Gradient Boosting Regressor...")
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=69)
gb.fit(X_train_processed, y_train)
y_pred_gb = gb.predict(X_test_processed)

gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_mse = mean_squared_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)

print(f"[OK] Gradient Boosting trained!")
print(f"     MAE: {gb_mae:.2f}")
print(f"     MSE: {gb_mse:.2f}")
print(f"     R²: {gb_r2:.4f}")

# Cell 11: Train XGBoost (if available)
if HAS_XGBOOST:
    print("\n[11/14] Training XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1,
                       objective='reg:squarederror', random_state=69)
    xgb.fit(X_train_processed, y_train)
    y_pred_xgb = xgb.predict(X_test_processed)

    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_mse = mean_squared_error(y_test, y_pred_xgb)
    xgb_r2 = r2_score(y_test, y_pred_xgb)

    print(f"[OK] XGBoost trained!")
    print(f"     MAE: {xgb_mae:.2f}")
    print(f"     MSE: {xgb_mse:.2f}")
    print(f"     R²: {xgb_r2:.4f}")
else:
    print("\n[11/14] Skipping XGBoost (not available)")
    xgb_mae = xgb_mse = xgb_r2 = 0

# Cell 12: Model Comparison
print("\n[12/14] Comparing model performance...")

models = ['Lasso Regression', 'Random Forest', 'Gradient Boosting']
mae_values = [lasso_mae, rf_mae, gb_mae]
mse_values = [lasso_mse, rf_mse, gb_mse]
r2_values = [lasso_r2, rf_r2, gb_r2]

if HAS_XGBOOST:
    models.append('XGBoost')
    mae_values.append(xgb_mae)
    mse_values.append(xgb_mse)
    r2_values.append(xgb_r2)

comparison_df = pd.DataFrame({
    'Model': models,
    'MAE': mae_values,
    'MSE': mse_values,
    'R²': r2_values
})

print("\n[INFO] Model Comparison Results:")
print(comparison_df.to_string(index=False))

# Identify best model
best_idx = comparison_df['R²'].idxmax()
best_model = comparison_df.loc[best_idx, 'Model']
print(f"\n[BEST MODEL] {best_model} (R² = {comparison_df.loc[best_idx, 'R²']:.4f})")

# Cell 13: Visualization 3 - Model Comparison
print("\n[13/14] Creating model comparison visualization...")

if USE_PLOTLY:
    comparison_viz = comparison_df.copy()
    comparison_viz['MAE_norm'] = comparison_viz['MAE'] / comparison_viz['MAE'].max()
    comparison_viz['MSE_norm'] = comparison_viz['MSE'] / comparison_viz['MSE'].max()
    comparison_viz['R²_norm'] = comparison_viz['R²'] / max(abs(comparison_viz['R²'].min()), comparison_viz['R²'].max())

    fig = go.Figure(data=[
        go.Bar(name='MAE', x=comparison_viz['Model'], y=comparison_viz['MAE_norm']),
        go.Bar(name='MSE', x=comparison_viz['Model'], y=comparison_viz['MSE_norm']),
        go.Bar(name='R²', x=comparison_viz['Model'], y=comparison_viz['R²_norm'])
    ])

    fig.update_layout(
        barmode='group',
        title='Model Performance Comparison (Normalized)',
        yaxis_title='Normalized Score',
        template='plotly_dark'
    )
    fig.write_html('viz/model_comparison.html')
    print(f"[OK] Saved: viz/model_comparison.html")
else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MAE
    axes[0].bar(comparison_df['Model'], comparison_df['MAE'], color='skyblue')
    axes[0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # MSE
    axes[1].bar(comparison_df['Model'], comparison_df['MSE'], color='lightcoral')
    axes[1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MSE', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    # R²
    axes[2].bar(comparison_df['Model'], comparison_df['R²'], color='lightgreen')
    axes[2].set_title('R² Score', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('viz/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: viz/model_comparison.png")

# Cell 14: Save Models
print("\n[14/14] Saving models and preprocessor...")

# Save best model (use XGBoost for deployment compatibility)
if HAS_XGBOOST:
    model_path = 'models/car_price_xgboost_model.joblib'
    joblib.dump(xgb, model_path)
    print(f"[OK] XGBoost model saved: {model_path}")
else:
    model_path = 'models/car_price_gradient_boosting_model.joblib'
    joblib.dump(gb, model_path)
    print(f"[OK] Gradient Boosting model saved: {model_path}")

# Save preprocessor
preprocessor_path = 'models/car_price_preprocessor.joblib'
joblib.dump(preprocessor, preprocessor_path)
print(f"[OK] Preprocessor saved: {preprocessor_path}")

# Save feature info
feature_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'all_features': X.columns.tolist()
}
feature_info_path = 'models/car_price_feature_info.joblib'
joblib.dump(feature_info, feature_info_path)
print(f"[OK] Feature info saved: {feature_info_path}")

# Summary
print("\n" + "="*80)
print("EXECUTION SUMMARY")
print("="*80)

# Count visualizations
viz_files = [f for f in os.listdir('viz') if f.endswith(('.html', '.png'))]
print(f"\n[VISUALIZATIONS] Generated {len(viz_files)} files in viz/:")
for f in sorted(viz_files):
    size = os.path.getsize(os.path.join('viz', f))
    print(f"  - {f} ({size:,} bytes)")

# Count models
model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
print(f"\n[MODELS] Saved {len(model_files)} files in models/:")
for f in sorted(model_files):
    size = os.path.getsize(os.path.join('models', f))
    print(f"  - {f} ({size:,} bytes)")

# Performance summary
print(f"\n[PERFORMANCE] Best Model: {best_model}")
print(f"  - MAE: {comparison_df.loc[best_idx, 'MAE']:,.2f}")
print(f"  - MSE: {comparison_df.loc[best_idx, 'MSE']:,.2f}")
print(f"  - R²: {comparison_df.loc[best_idx, 'R²']:.4f}")

print("\n" + "="*80)
print("ALL CELLS EXECUTED SUCCESSFULLY!")
print("="*80)
