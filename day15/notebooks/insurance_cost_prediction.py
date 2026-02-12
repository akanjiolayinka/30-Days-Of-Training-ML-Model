"""
Day 15: Medical Insurance Cost Prediction
Predict insurance charges based on demographic and health factors.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'viz')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=== Medical Insurance Cost Prediction ===")
df = pd.read_csv(os.path.join(DATA_DIR, 'insurance.csv'))
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nTarget (charges) statistics:")
print(df['charges'].describe())

# === DATA PREPROCESSING ===
print("\n=== Data Preprocessing ===")

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex_encoded'] = le_sex.fit_transform(df['sex'])
df['smoker_encoded'] = le_smoker.fit_transform(df['smoker'])
df['region_encoded'] = le_region.fit_transform(df['region'])

print(f"Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"Smoker encoding: {dict(zip(le_smoker.classes_, le_smoker.transform(le_smoker.classes_)))}")
print(f"Region encoding: {dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))}")

# === VISUALIZATIONS ===
print("\n=== Generating Visualizations ===")

# 1. Charges distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['charges'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Insurance Charges ($)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Insurance Charges')

axes[1].hist(np.log1p(df['charges']), bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Log(Insurance Charges)')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Log-Transformed Charges')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'charges_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/charges_distribution.png")

# 2. Charges by categorical features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(data=df, x='sex', y='charges', palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Insurance Charges by Sex')
axes[0, 0].set_ylabel('Charges ($)')

sns.boxplot(data=df, x='smoker', y='charges', palette=['#2ecc71', '#e74c3c'], ax=axes[0, 1])
axes[0, 1].set_title('Insurance Charges by Smoker Status')
axes[0, 1].set_ylabel('Charges ($)')

sns.boxplot(data=df, x='children', y='charges', palette='Set3', ax=axes[1, 0])
axes[1, 0].set_title('Insurance Charges by Number of Children')
axes[1, 0].set_ylabel('Charges ($)')

sns.boxplot(data=df, x='region', y='charges', palette='Set1', ax=axes[1, 1])
axes[1, 1].set_title('Insurance Charges by Region')
axes[1, 1].set_ylabel('Charges ($)')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'charges_by_categories.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/charges_by_categories.png")

# 3. Continuous features vs charges
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(df['age'], df['charges'], alpha=0.5, s=20, color='#3498db')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Charges ($)')
axes[0, 0].set_title('Age vs Insurance Charges')

axes[0, 1].scatter(df['bmi'], df['charges'], alpha=0.5, s=20, color='#e74c3c')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('Charges ($)')
axes[0, 1].set_title('BMI vs Insurance Charges')

# Color by smoker
colors = ['#2ecc71' if s == 'no' else '#e74c3c' for s in df['smoker']]
axes[1, 0].scatter(df['age'], df['charges'], c=colors, alpha=0.5, s=20)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Charges ($)')
axes[1, 0].set_title('Age vs Charges (Green=Non-smoker, Red=Smoker)')

axes[1, 1].scatter(df['bmi'], df['charges'], c=colors, alpha=0.5, s=20)
axes[1, 1].set_xlabel('BMI')
axes[1, 1].set_ylabel('Charges ($)')
axes[1, 1].set_title('BMI vs Charges (Green=Non-smoker, Red=Smoker)')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'continuous_features_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/continuous_features_scatter.png")

# 4. Correlation heatmap
corr_features = ['age', 'bmi', 'children', 'charges', 'sex_encoded', 'smoker_encoded', 'region_encoded']
corr = df[corr_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/correlation_heatmap.png")

# === MODELING ===
print("\n=== Training Regression Models ===")

feature_cols = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded']
X = df[feature_cols]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Model 1: Linear Regression
print("\n--- Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
print(f"R²: {lr_r2:.4f}, RMSE: ${lr_rmse:.2f}, MAE: ${lr_mae:.2f}")

# Model 2: Ridge Regression
print("\n--- Ridge Regression ---")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)
print(f"R²: {ridge_r2:.4f}, RMSE: ${ridge_rmse:.2f}, MAE: ${ridge_mae:.2f}")

# Model 3: Random Forest
print("\n--- Random Forest Regressor ---")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"R²: {rf_r2:.4f}, RMSE: ${rf_rmse:.2f}, MAE: ${rf_mae:.2f}")

# Model comparison
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
    'R²': [lr_r2, ridge_r2, rf_r2],
    'RMSE': [lr_rmse, ridge_rmse, rf_rmse],
    'MAE': [lr_mae, ridge_mae, rf_mae]
})

print("\n=== Model Comparison ===")
print(results.to_string(index=False))

# 5. Model comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['R²', 'RMSE', 'MAE']
for idx, metric in enumerate(metrics):
    axes[idx].bar(results['Model'], results[metric], color=['#3498db','#2ecc71','#9b59b6'], edgecolor='black')
    axes[idx].set_title(f'Model Comparison: {metric}')
    axes[idx].set_ylabel(metric)
    axes[idx].tick_params(axis='x', rotation=15)
    for i, v in enumerate(results[metric]):
        if metric == 'R²':
            axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[idx].text(i, v, f'${v:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/model_comparison.png")

# 6. Actual vs Predicted (best model - Random Forest)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.5, s=30, color='#3498db')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.title(f'Random Forest: Actual vs Predicted (R² = {rf_r2:.4f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/actual_vs_predicted.png")

# 7. Residual plot
residuals = y_test - rf_pred
plt.figure(figsize=(10, 6))
plt.scatter(rf_pred, residuals, alpha=0.5, s=30, color='#e74c3c')
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.xlabel('Predicted Charges ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot (Random Forest)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'residual_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/residual_plot.png")

# 8. Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='#2ecc71', edgecolor='black')
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_importance.png")

# === SAVE MODELS ===
print("\n=== Saving Models ===")
joblib.dump(lr, os.path.join(MODELS_DIR, 'linear_regression_model.joblib'))
joblib.dump(ridge, os.path.join(MODELS_DIR, 'ridge_regression_model.joblib'))
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(le_sex, os.path.join(MODELS_DIR, 'sex_encoder.joblib'))
joblib.dump(le_smoker, os.path.join(MODELS_DIR, 'smoker_encoder.joblib'))
joblib.dump(le_region, os.path.join(MODELS_DIR, 'region_encoder.joblib'))

print("  linear_regression_model.joblib")
print("  ridge_regression_model.joblib")
print("  random_forest_model.joblib")
print("  scaler.joblib")
print("  sex_encoder.joblib")
print("  smoker_encoder.joblib")
print("  region_encoder.joblib")

print("\n=== Done! ===")
print(f"Best model: Random Forest (R² = {rf_r2:.4f}, RMSE = ${rf_rmse:.2f})")
