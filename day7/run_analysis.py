"""
Day 7: Fruit Classification - Execute Analysis Cell by Cell (Matplotlib Version)
This script runs the complete analysis with visualization saving using matplotlib
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("DAY 7: FRUIT CLASSIFICATION - STEP-BY-STEP EXECUTION")
print("="*70)

# Create directories
os.makedirs('viz', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================================
# CELL 1: Import Libraries
# ============================================================================
print("\n[CELL 1] Importing libraries...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("[OK] All libraries imported successfully")
    print(f"[OK] Visualization directory: ./viz")
    print("[NOTE] Using matplotlib for visualizations (Plotly not available)")
except ImportError as e:
    print(f"[ERROR] Missing library: {e}")
    exit(1)

# ============================================================================
# CELL 2: Load Dataset
# ============================================================================
print("\n[CELL 2] Loading dataset...")
fruit_data = pd.read_csv('data/fruit_classification_dataset.csv')
print(f"[OK] Dataset loaded successfully!")
print(f"[OK] Shape: {fruit_data.shape}")
print("\nFirst 5 rows:")
print(fruit_data.head())

# ============================================================================
# CELL 3: Explore Dataset
# ============================================================================
print("\n[CELL 3] Exploring dataset...")
print("\nDataset Info:")
fruit_data.info()
print("\nMissing Values:")
print(fruit_data.isnull().sum())
print("\nUnique Fruit Types:")
print(fruit_data['fruit_name'].value_counts())
print("\nDescriptive Statistics:")
print(fruit_data.describe())

# ============================================================================
# CELL 4: Visualize Numerical Features
# ============================================================================
print("\n[CELL 4] Creating numerical distributions visualization...")
numerical_features = ['size (cm)', 'weight (g)', 'avg_price (₹)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, feature in enumerate(numerical_features):
    axes[i].hist(fruit_data[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viz/numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/numerical_distributions.png")

# ============================================================================
# CELL 5: Individual Feature Histograms by Fruit
# ============================================================================
print("\n[CELL 5] Creating feature-by-fruit distributions...")

for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    for fruit in fruit_data['fruit_name'].unique():
        fruit_subset = fruit_data[fruit_data['fruit_name'] == fruit]
        plt.hist(fruit_subset[feature], bins=20, alpha=0.3, label=fruit)

    plt.title(f'Distribution of {feature} by Fruit Type')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    filename = f"viz/{feature.replace(' ', '_').replace('(', '').replace(')', '').replace('₹', 'inr')}_by_fruit.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filename}")

# ============================================================================
# CELL 6: Correlation Heatmap
# ============================================================================
print("\n[CELL 6] Creating correlation heatmap...")
corr_matrix = fruit_data[numerical_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.3f', square=True, linewidths=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('viz/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/correlation_heatmap.png")
print("\nCorrelation Matrix:")
print(corr_matrix)

# ============================================================================
# CELL 7: Average Price by Fruit
# ============================================================================
print("\n[CELL 7] Creating average price by fruit visualization...")
avg_price_by_fruit = fruit_data.groupby('fruit_name')['avg_price (₹)'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
avg_price_by_fruit.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Average Price by Fruit Type')
plt.xlabel('Fruit')
plt.ylabel('Average Price (₹)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('viz/avg_price_by_fruit.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/avg_price_by_fruit.png")
print("\nTop 5 Most Expensive Fruits:")
print(avg_price_by_fruit.head())

# ============================================================================
# CELL 8: Size vs Weight Scatter Plot
# ============================================================================
print("\n[CELL 8] Creating size vs weight scatter plot...")
plt.figure(figsize=(14, 8))

# Create a color map for fruits
fruits = fruit_data['fruit_name'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(fruits)))
fruit_color_map = dict(zip(fruits, colors))

for fruit in fruits:
    fruit_subset = fruit_data[fruit_data['fruit_name'] == fruit]
    plt.scatter(fruit_subset['size (cm)'],
               fruit_subset['weight (g)'],
               s=fruit_subset['avg_price (₹)'] * 2,  # size based on price
               alpha=0.6,
               label=fruit,
               color=fruit_color_map[fruit])

plt.title('Size vs Weight Scatter Plot (Bubble Size = Price)')
plt.xlabel('Size (cm)')
plt.ylabel('Weight (g)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('viz/size_vs_weight_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/size_vs_weight_scatter.png")

# ============================================================================
# CELL 9: Taste by Color
# ============================================================================
print("\n[CELL 9] Creating taste by color visualization...")
taste_color_counts = pd.crosstab(fruit_data['taste'], fruit_data['color'])

taste_color_counts.plot(kind='bar', figsize=(10, 6), edgecolor='black')
plt.title('Taste Distribution by Color')
plt.xlabel('Taste')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Color', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('viz/taste_by_color.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/taste_by_color.png")

# ============================================================================
# CELL 10: Encode Categorical Features
# ============================================================================
print("\n[CELL 10] Encoding categorical features...")
encoded_data = fruit_data.copy()

le_target = LabelEncoder()
encoded_data['fruit_name_encoded'] = le_target.fit_transform(encoded_data['fruit_name'])

categorical_features = ['shape', 'color', 'taste']
encoded_data = pd.get_dummies(encoded_data, columns=categorical_features, drop_first=True)

print(f"[OK] Features encoded: {len(encoded_data.columns)} total features")

# ============================================================================
# CELL 11: Split Data
# ============================================================================
print("\n[CELL 11] Splitting data...")
X = encoded_data.drop(['fruit_name', 'fruit_name_encoded'], axis=1)
y = encoded_data['fruit_name_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[OK] Training set shape: {X_train.shape}")
print(f"[OK] Testing set shape: {X_test.shape}")
print(f"[OK] Number of classes: {len(le_target.classes_)}")

# ============================================================================
# CELL 12: Train Random Forest Model
# ============================================================================
print("\n[CELL 12] Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print(f"[OK] Random Forest model trained successfully!")
print(f"[OK] Number of estimators: {rf_model.n_estimators}")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

# ============================================================================
# CELL 13: Feature Importance Visualization
# ============================================================================
print("\n[CELL 13] Creating feature importance visualization...")
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('viz/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/feature_importance.png")

# ============================================================================
# CELL 14: Evaluate Model
# ============================================================================
print("\n[CELL 14] Evaluating model...")
y_pred = rf_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

accuracy = rf_model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# ============================================================================
# CELL 15: Confusion Matrix
# ============================================================================
print("\n[CELL 15] Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('viz/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] viz/confusion_matrix.png")

# ============================================================================
# CELL 16: Save Model
# ============================================================================
print("\n[CELL 16] Saving model...")
import joblib

joblib.dump(rf_model, 'models/fruit_rf_model.joblib')
joblib.dump(le_target, 'models/label_encoder.joblib')

print("[SAVED] models/fruit_rf_model.joblib")
print("[SAVED] models/label_encoder.joblib")

# ============================================================================
# CELL 17: Test Prediction
# ============================================================================
print("\n[CELL 17] Testing prediction on sample data...")
sample_data = pd.DataFrame([{
    'size (cm)': 25.0,
    'weight (g)': 3000.0,
    'avg_price (₹)': 140.0,
    'shape_oval': 0,
    'shape_round': 1,
    'color_brown': 0,
    'color_green': 1,
    'color_orange': 0,
    'color_pink': 0,
    'color_purple': 0,
    'color_red': 0,
    'color_yellow': 0,
    'taste_sweet': 1,
    'taste_tangy': 0
}])

sample_data = sample_data[X.columns]

prediction = rf_model.predict(sample_data)
predicted_fruit = le_target.inverse_transform(prediction)[0]
prediction_proba = rf_model.predict_proba(sample_data)[0]
confidence = prediction_proba.max() * 100

print("="*60)
print("PREDICTION TEST")
print("="*60)
print(f"Sample: Size=25cm, Weight=3000g, Price=Rs.140, Round, Green, Sweet")
print(f"\nPredicted Fruit: {predicted_fruit.upper()}")
print(f"Confidence: {confidence:.2f}%")
print("="*60)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EXECUTION COMPLETE - SUMMARY")
print("="*70)

# Check visualization files
viz_files = [
    'numerical_distributions.png',
    'size_cm_by_fruit.png',
    'weight_g_by_fruit.png',
    'avg_price_inr_by_fruit.png',
    'correlation_heatmap.png',
    'avg_price_by_fruit.png',
    'size_vs_weight_scatter.png',
    'taste_by_color.png',
    'feature_importance.png',
    'confusion_matrix.png'
]

print("\nVISUALIZATIONS SAVED:")
viz_count = 0
for viz_file in viz_files:
    path = f'viz/{viz_file}'
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  [OK] {viz_file} ({size:,} bytes)")
        viz_count += 1
    else:
        print(f"  [MISSING] {viz_file}")

print(f"\nTotal visualizations saved: {viz_count}/{len(viz_files)}")

print("\nMODELS SAVED:")
if os.path.exists('models/fruit_rf_model.joblib'):
    model_size = os.path.getsize('models/fruit_rf_model.joblib')
    print(f"  [OK] fruit_rf_model.joblib ({model_size:,} bytes)")
if os.path.exists('models/label_encoder.joblib'):
    encoder_size = os.path.getsize('models/label_encoder.joblib')
    print(f"  [OK] label_encoder.joblib ({encoder_size:,} bytes)")

print("\nKEY RESULTS:")
print(f"  Dataset: {fruit_data.shape[0]} samples, {fruit_data['fruit_name'].nunique()} fruit types")
print(f"  Features: {X.shape[1]} (after encoding)")
print(f"  Model Accuracy: {accuracy*100:.2f}%")
print(f"  Total Cells Executed: 17")

print("\n" + "="*70)
print("ALL CELLS EXECUTED SUCCESSFULLY!")
print("="*70)
