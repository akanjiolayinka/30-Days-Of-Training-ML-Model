"""
Day 23: Advanced Multi-Domain Dataset Analysis with Ensemble ML
Implements sophisticated predictive models with feature engineering.
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

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Try to import SHAP for advanced interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - skipping interpretability analysis")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'viz')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=== Loading Unified Multi-Domain Dataset ===")
df = pd.read_csv(os.path.join(DATA_DIR, 'UnifiedDataset.csv'))
print(f"Shape: {df.shape}")
print(f"Columns: {df.shape[1]} features")

# === DATA PREPROCESSING ===
print("\n=== Advanced Data Preprocessing ===")

# Select numeric columns for modeling (exclude identifiers and extreme sparsity)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove columns with >50% missing values
valid_cols = [col for col in numeric_cols if df[col].isnull().sum() / len(df) < 0.5]

# Target: Life Expectancy (regression task)
target = 'Life Expectancy'
if target in valid_cols:
    valid_cols.remove(target)

# Create modeling dataset
model_df = df[[target] + valid_cols].copy()
model_df = model_df.dropna(subset=[target])  # Must have target

print(f"Features after filtering: {len(valid_cols)}")
print(f"Samples: {len(model_df)}")
print(f"Target (Life Expectancy) range: {model_df[target].min():.1f} - {model_df[target].max():.1f}")

# Impute missing values with median
for col in valid_cols:
    if model_df[col].isnull().sum() > 0:
        model_df[col].fillna(model_df[col].median(), inplace=True)

# === ADVANCED FEATURE ENGINEERING ===
print("\n=== Feature Engineering ===")

# Select most important features using statistical test
selector = SelectKBest(score_func=f_regression, k=min(30, len(valid_cols)))
X = model_df[valid_cols]
y = model_df[target]

selector.fit(X, y)
selected_features = [valid_cols[i] for i in selector.get_support(indices=True)]
print(f"Selected top 30 features using SelectKBest")

# Create polynomial features for top 5 most important
top_5 = selected_features[:5]
print(f"Creating interaction terms for top 5 features: {top_5}")

for i, feat1 in enumerate(top_5):
    # Square terms
    model_df[f'{feat1}_squared'] = model_df[feat1] ** 2
    # Interaction terms
    for feat2 in top_5[i+1:]:
        model_df[f'{feat1}_x_{feat2}'] = model_df[feat1] * model_df[feat2]

# Update feature list
engineered_features = [col for col in model_df.columns if col != target]
print(f"Total features after engineering: {len(engineered_features)}")

# === TRAIN-TEST SPLIT ===
X = model_df[engineered_features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# === MODEL TRAINING ===
print("\n=== Training Ensemble Models ===")

# Model 1: Random Forest with hyperparameter tuning
print("\n--- Random Forest Regressor ---")
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)

print(f"Best RF parameters: {rf_grid.best_params_}")
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"RF R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")

# Model 2: Gradient Boosting Regressor
print("\n--- Gradient Boosting Regressor ---")
gb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
gb = GradientBoostingRegressor(random_state=42)
gb_grid = GridSearchCV(gb, gb_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
gb_grid.fit(X_train, y_train)

print(f"Best GB parameters: {gb_grid.best_params_}")
gb_best = gb_grid.best_estimator_
gb_pred = gb_best.predict(X_test)
gb_r2 = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mae = mean_absolute_error(y_test, gb_pred)
print(f"GB R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}")

# Model 3: Voting Ensemble (combining RF + GB)
print("\n--- Voting Ensemble ---")
voting = VotingRegressor([
    ('rf', rf_best),
    ('gb', gb_best)
])
voting.fit(X_train, y_train)
voting_pred = voting.predict(X_test)
voting_r2 = r2_score(y_test, voting_pred)
voting_rmse = np.sqrt(mean_squared_error(y_test, voting_pred))
voting_mae = mean_absolute_error(y_test, voting_pred)
print(f"Voting R²: {voting_r2:.4f}, RMSE: {voting_rmse:.4f}, MAE: {voting_mae:.4f}")

# === MODEL COMPARISON ===
results = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Voting Ensemble'],
    'R²': [rf_r2, gb_r2, voting_r2],
    'RMSE': [rf_rmse, gb_rmse, voting_rmse],
    'MAE': [rf_mae, gb_mae, voting_mae]
})

print("\n=== Model Comparison ===")
print(results.to_string(index=False))

# === VISUALIZATIONS ===
print("\n=== Generating Visualizations ===")

# 1. Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['R²', 'RMSE', 'MAE']
for idx, metric in enumerate(metrics):
    axes[idx].bar(results['Model'], results[metric], color=['#3498db','#2ecc71','#9b59b6'], edgecolor='black')
    axes[idx].set_title(f'Model Comparison: {metric}')
    axes[idx].set_ylabel(metric)
    axes[idx].tick_params(axis='x', rotation=15)
    for i, v in enumerate(results[metric]):
        axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/model_comparison.png")

# 2. Actual vs Predicted (best model - voting)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, voting_pred, alpha=0.5, s=20, color='#3498db')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title(f'Voting Ensemble: Actual vs Predicted (R² = {voting_r2:.4f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/actual_vs_predicted.png")

# 3. Residual Plot
residuals = y_test - voting_pred
plt.figure(figsize=(10, 6))
plt.scatter(voting_pred, residuals, alpha=0.5, s=20, color='#e74c3c')
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.xlabel('Predicted Life Expectancy')
plt.ylabel('Residuals')
plt.title('Residual Plot (Voting Ensemble)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'residual_plot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/residual_plot.png")

# 4. Feature Importance (Random Forest)
importances = rf_best.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20
top_features = [engineered_features[i] for i in indices]
top_importances = importances[indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_importances, color='#2ecc71', edgecolor='black')
plt.yticks(range(len(top_features)), top_features, fontsize=8)
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance_rf.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_importance_rf.png")

# 5. Feature Importance (Gradient Boosting)
gb_importances = gb_best.feature_importances_
gb_indices = np.argsort(gb_importances)[-20:]
gb_top_features = [engineered_features[i] for i in gb_indices]
gb_top_importances = gb_importances[gb_indices]

plt.figure(figsize=(10, 8))
plt.barh(range(len(gb_top_features)), gb_top_importances, color='#f39c12', edgecolor='black')
plt.yticks(range(len(gb_top_features)), gb_top_features, fontsize=8)
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances (Gradient Boosting)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance_gb.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_importance_gb.png")

# 6. Learning Curve visualization
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
train_scores = []
test_scores = []

for size in train_sizes:
    n_samples = int(len(X_train) * size)
    rf_temp = RandomForestRegressor(**rf_best.get_params())
    rf_temp.fit(X_train[:n_samples], y_train[:n_samples])
    train_scores.append(r2_score(y_train[:n_samples], rf_temp.predict(X_train[:n_samples])))
    test_scores.append(r2_score(y_test, rf_temp.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot([s*100 for s in train_sizes], train_scores, 'o-', color='#2ecc71', label='Training Score', linewidth=2)
plt.plot([s*100 for s in train_sizes], test_scores, 'o-', color='#e74c3c', label='Test Score', linewidth=2)
plt.xlabel('Training Set Size (%)')
plt.ylabel('R² Score')
plt.title('Learning Curve (Random Forest)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'learning_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/learning_curve.png")

# 7. SHAP Analysis (if available)
if SHAP_AVAILABLE:
    try:
        print("\n=== SHAP Analysis ===")
        explainer = shap.TreeExplainer(rf_best)
        shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:100], plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: viz/shap_summary.png")
    except Exception as e:
        print(f"  SHAP visualization failed: {e}")

# === SAVE MODELS ===
print("\n=== Saving Models ===")
joblib.dump(rf_best, os.path.join(MODELS_DIR, 'random_forest_model.joblib'))
joblib.dump(gb_best, os.path.join(MODELS_DIR, 'gradient_boosting_model.joblib'))
joblib.dump(voting, os.path.join(MODELS_DIR, 'voting_ensemble_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'feature_scaler.joblib'))

# Save feature names
with open(os.path.join(MODELS_DIR, 'feature_names.txt'), 'w') as f:
    f.write('\n'.join(engineered_features))

print("  random_forest_model.joblib")
print("  gradient_boosting_model.joblib")
print("  voting_ensemble_model.joblib")
print("  feature_scaler.joblib")
print("  feature_names.txt")

print("\n=== Analysis Complete ===")
print(f"Best model: Voting Ensemble (R² = {voting_r2:.4f})")
print(f"Target prediction range: {y.min():.1f} - {y.max():.1f} years")
