"""
Day 12: Wine Quality Classification
Predict wine quality based on physicochemical properties.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'viz')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=== Wine Quality Classification Analysis ===")
df = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'))
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Check quality distribution
print(f"\nQuality distribution:\n{df['quality'].value_counts().sort_index()}")

# Create binary classification: Good (>=6) vs Not Good (<6)
df['quality_binary'] = (df['quality'] >= 6).astype(int)
print(f"\nBinary classification:")
print(f"Not Good (quality < 6): {(df['quality_binary'] == 0).sum()}")
print(f"Good (quality >= 6): {(df['quality_binary'] == 1).sum()}")

# === VISUALIZATIONS ===
print("\n=== Generating Visualizations ===")

# 1. Quality distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
quality_counts = df['quality'].value_counts().sort_index()
axes[0].bar(quality_counts.index, quality_counts.values, color='#8e44ad', edgecolor='black')
axes[0].set_xlabel('Quality Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Wine Quality Distribution')

binary_counts = df['quality_binary'].value_counts()
axes[1].bar(['Not Good (<6)', 'Good (≥6)'], binary_counts.values, color=['#e74c3c','#27ae60'], edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('Binary Quality Classification')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'quality_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/quality_distribution.png")

# 2. Feature distributions
features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feat in enumerate(features):
    ax = axes[idx//2, idx%2]
    df.boxplot(column=feat, by='quality_binary', ax=ax)
    ax.set_title(f'{feat.title()} by Quality')
    ax.set_xlabel('Quality (0=Not Good, 1=Good)')
    ax.set_ylabel(feat.title())
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_boxplots.png")

# 3. Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('quality_binary')
corr = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/correlation_heatmap.png")

# 4. Alcohol vs Quality scatter
plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if q < 6 else '#27ae60' for q in df['quality']]
plt.scatter(df['alcohol'], df['volatile acidity'], c=colors, alpha=0.5, s=30)
plt.xlabel('Alcohol %')
plt.ylabel('Volatile Acidity')
plt.title('Alcohol vs Volatile Acidity (Red=Not Good, Green=Good)')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Not Good (<6)'),
                   Patch(facecolor='#27ae60', label='Good (≥6)')]
plt.legend(handles=legend_elements)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'alcohol_vs_acidity_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/alcohol_vs_acidity_scatter.png")

# === MODELING ===
print("\n=== Training Classification Models ===")

feature_cols = [col for col in df.columns if col not in ['quality', 'quality_binary']]
X = df[feature_cols]
y = df['quality_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Model 1: Logistic Regression
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_acc:.4f}")
print(classification_report(y_test, lr_pred, target_names=['Not Good', 'Good']))

# Model 2: Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred, target_names=['Not Good', 'Good']))

# Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_acc, rf_acc]
axes[0].bar(models, accuracies, color=['#3498db','#2ecc71'], edgecolor='black')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Comparison')
axes[0].set_ylim([0, 1])
for i, v in enumerate(accuracies):
    axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom')

# Confusion matrix for best model
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Not Good', 'Good'], yticklabels=['Not Good', 'Good'])
axes[1].set_title('Confusion Matrix (Random Forest)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'model_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/model_performance.png")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='#9b59b6', edgecolor='black')
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_importance.png")

# === SAVE MODELS ===
print("\n=== Saving Models ===")
joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression_model.joblib'))
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

print("  logistic_regression_model.joblib")
print("  random_forest_model.joblib")
print("  scaler.joblib")

print("\n=== Done! ===")
print(f"Best model: Random Forest (Accuracy: {rf_acc:.4f})")
