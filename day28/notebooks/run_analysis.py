"""
Day 28: Fitness & Nutrition Tracking Analysis
Analyze hybrid dataset combining workout metrics with meal tracking.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'viz')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=== Loading Fitness + Nutrition Dataset ===")
df = pd.read_csv(os.path.join(DATA_DIR, 'Final_data.csv'))
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns[:20])}... ({len(df.columns)} total)")

# Data cleaning
print("\n=== Data Cleaning ===")
# Select key columns for analysis
analysis_cols = ['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI', 'Max_BPM', 'Avg_BPM',
                  'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Workout_Type',
                  'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
                  'Experience_Level', 'Carbs', 'Proteins', 'Fats', 'Calories', 'meal_type',
                  'diet_type', 'cal_balance', 'pct_carbs', 'protein_per_kg']

df_clean = df[analysis_cols].copy()
df_clean = df_clean.dropna(subset=['Workout_Type', 'meal_type', 'diet_type'])

print(f"Cleaned shape: {df_clean.shape}")
print(f"\nWorkout types: {df_clean['Workout_Type'].value_counts().to_dict()}")
print(f"Meal types: {df_clean['meal_type'].value_counts().to_dict()}")
print(f"Diet types: {df_clean['diet_type'].value_counts().to_dict()}")

# === VISUALIZATIONS ===
print("\n=== Generating Visualizations ===")

# 1. Workout Type Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
workout_counts = df_clean['Workout_Type'].value_counts()
axes[0].bar(workout_counts.index, workout_counts.values, color=['#e74c3c','#3498db','#2ecc71','#f39c12'], edgecolor='black')
axes[0].set_title('Workout Type Distribution')
axes[0].set_xlabel('Workout Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)

diet_counts = df_clean['diet_type'].value_counts()
axes[1].bar(diet_counts.index, diet_counts.values, color=['#9b59b6','#1abc9c','#e67e22','#34495e'], edgecolor='black')
axes[1].set_title('Diet Type Distribution')
axes[1].set_xlabel('Diet Type')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'workout_diet_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/workout_diet_distribution.png")

# 2. Calories Burned by Workout Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='Workout_Type', y='Calories_Burned', palette='Set2')
plt.title('Calories Burned by Workout Type')
plt.xlabel('Workout Type')
plt.ylabel('Calories Burned')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'calories_by_workout.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/calories_by_workout.png")

# 3. BMI vs Workout Frequency
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df_clean['Workout_Frequency (days/week)'], df_clean['BMI'], alpha=0.3, s=20, color='#e74c3c')
axes[0].set_title('BMI vs Workout Frequency')
axes[0].set_xlabel('Workout Frequency (days/week)')
axes[0].set_ylabel('BMI')

axes[1].scatter(df_clean['Water_Intake (liters)'], df_clean['Fat_Percentage'], alpha=0.3, s=20, color='#3498db')
axes[1].set_title('Fat Percentage vs Water Intake')
axes[1].set_xlabel('Water Intake (liters)')
axes[1].set_ylabel('Fat Percentage (%)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'fitness_metrics_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/fitness_metrics_scatter.png")

# 4. Calorie Balance Analysis
plt.figure(figsize=(10, 6))
cal_balance_bins = pd.cut(df_clean['cal_balance'], bins=[-np.inf, -500, 0, 500, np.inf],
                            labels=['Large Deficit', 'Small Deficit', 'Small Surplus', 'Large Surplus'])
cal_counts = cal_balance_bins.value_counts().sort_index()
plt.bar(cal_counts.index.astype(str), cal_counts.values, color=['#c0392b','#e67e22','#f1c40f','#27ae60'], edgecolor='black')
plt.title('Calorie Balance Distribution')
plt.xlabel('Calorie Balance Category')
plt.ylabel('Count')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'calorie_balance_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/calorie_balance_distribution.png")

# 5. Meal Type vs Macros
meal_macros = df_clean.groupby('meal_type')[['Carbs', 'Proteins', 'Fats']].mean()
meal_macros.plot(kind='bar', figsize=(10, 6), color=['#f39c12','#e74c3c','#3498db'], edgecolor='black')
plt.title('Average Macronutrients by Meal Type')
plt.xlabel('Meal Type')
plt.ylabel('Grams')
plt.xticks(rotation=0)
plt.legend(title='Macronutrient')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'meal_type_macros.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/meal_type_macros.png")

# 6. Heart Rate Analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(df_clean['Resting_BPM'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[0].set_title('Resting Heart Rate Distribution')
axes[0].set_xlabel('Resting BPM')
axes[0].set_ylabel('Count')

axes[1].hist(df_clean['Avg_BPM'], bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
axes[1].set_title('Average Heart Rate Distribution')
axes[1].set_xlabel('Average BPM')
axes[1].set_ylabel('Count')

axes[2].hist(df_clean['Max_BPM'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[2].set_title('Max Heart Rate Distribution')
axes[2].set_xlabel('Max BPM')
axes[2].set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'heart_rate_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/heart_rate_distributions.png")

# 7. Correlation Heatmap
numeric_cols = ['Age', 'Weight (kg)', 'BMI', 'Avg_BPM', 'Calories_Burned', 'Fat_Percentage',
                'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Carbs', 'Proteins', 'Fats', 'cal_balance']
corr = df_clean[numeric_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Fitness & Nutrition Metrics')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/correlation_heatmap.png")

# === CLASSIFICATION MODELS ===
print("\n=== Training Classification Models ===")

# Model 1: Predict Workout Type (based on fitness metrics)
print("\n--- Model 1: Workout Type Classification ---")
feature_cols_workout = ['Age', 'BMI', 'Resting_BPM', 'Avg_BPM', 'Max_BPM',
                         'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)']
workout_df = df_clean[feature_cols_workout + ['Workout_Type']].dropna()

le_workout = LabelEncoder()
workout_df = workout_df.copy()
workout_df['target'] = le_workout.fit_transform(workout_df['Workout_Type'])

X_w = workout_df[feature_cols_workout]
y_w = workout_df['target']

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w, test_size=0.2, random_state=42, stratify=y_w)

scaler_workout = StandardScaler()
X_w_train_scaled = scaler_workout.fit_transform(X_w_train)
X_w_test_scaled = scaler_workout.transform(X_w_test)

rf_workout = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_workout.fit(X_w_train, y_w_train)
y_w_pred = rf_workout.predict(X_w_test)

print(f"Accuracy: {accuracy_score(y_w_test, y_w_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_w_test, y_w_pred, target_names=le_workout.classes_))

# Model 2: Predict Diet Type (based on nutrition + fitness metrics)
print("\n--- Model 2: Diet Type Classification ---")
feature_cols_diet = ['Age', 'BMI', 'Calories_Burned', 'Carbs', 'Proteins', 'Fats',
                     'Calories', 'pct_carbs', 'protein_per_kg', 'cal_balance']
diet_df = df_clean[feature_cols_diet + ['diet_type']].dropna()

le_diet = LabelEncoder()
diet_df = diet_df.copy()
diet_df['target'] = le_diet.fit_transform(diet_df['diet_type'])

X_d = diet_df[feature_cols_diet]
y_d = diet_df['target']

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size=0.2, random_state=42, stratify=y_d)

scaler_diet = StandardScaler()
X_d_train_scaled = scaler_diet.fit_transform(X_d_train)
X_d_test_scaled = scaler_diet.transform(X_d_test)

rf_diet = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_diet.fit(X_d_train, y_d_train)
y_d_pred = rf_diet.predict(X_d_test)

print(f"Accuracy: {accuracy_score(y_d_test, y_d_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_d_test, y_d_pred, target_names=le_diet.classes_))

# Confusion matrices visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm1 = confusion_matrix(y_w_test, y_w_pred)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=le_workout.classes_, yticklabels=le_workout.classes_)
axes[0].set_title('Confusion Matrix: Workout Type')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm2 = confusion_matrix(y_d_test, y_d_pred)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=le_diet.classes_, yticklabels=le_diet.classes_)
axes[1].set_title('Confusion Matrix: Diet Type')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/confusion_matrices.png")

# Feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
importances1 = pd.Series(rf_workout.feature_importances_, index=feature_cols_workout).sort_values()
importances1.plot(kind='barh', ax=axes[0], color='#3498db', edgecolor='black')
axes[0].set_title('Feature Importance: Workout Type Prediction')
axes[0].set_xlabel('Importance')

importances2 = pd.Series(rf_diet.feature_importances_, index=feature_cols_diet).sort_values()
importances2.plot(kind='barh', ax=axes[1], color='#2ecc71', edgecolor='black')
axes[1].set_title('Feature Importance: Diet Type Prediction')
axes[1].set_xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_importance.png")

# === SAVE MODELS ===
print("\n=== Saving Models ===")
joblib.dump(rf_workout, os.path.join(MODELS_DIR, 'workout_type_rf_model.joblib'))
joblib.dump(rf_diet, os.path.join(MODELS_DIR, 'diet_type_rf_model.joblib'))
joblib.dump(scaler_workout, os.path.join(MODELS_DIR, 'workout_scaler.joblib'))
joblib.dump(scaler_diet, os.path.join(MODELS_DIR, 'diet_scaler.joblib'))
joblib.dump(le_workout, os.path.join(MODELS_DIR, 'workout_label_encoder.joblib'))
joblib.dump(le_diet, os.path.join(MODELS_DIR, 'diet_label_encoder.joblib'))

print("  workout_type_rf_model.joblib")
print("  diet_type_rf_model.joblib")
print("  workout_scaler.joblib")
print("  diet_scaler.joblib")
print("  workout_label_encoder.joblib")
print("  diet_label_encoder.joblib")

print("\n=== Done! ===")
