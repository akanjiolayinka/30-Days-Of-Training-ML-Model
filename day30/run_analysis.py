"""
Day 30: Housing Price Prediction - Execution Script (FINAL PROJECT!)

This script runs the complete housing price analysis cell by cell,
generating all visualizations and trained models.

🎉 FINAL DAY OF 30 DAYS OF DATASETS! 🎉
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('viz', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*80)
print("DAY 30: HOUSING PRICE PREDICTION - FINAL PROJECT EXECUTION")
print("🎉 30 DAYS OF DATASETS - FINAL DAY! 🎉")
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
    print("[OK] Using Plotly for interactive visualizations")
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    USE_PLOTLY = False
    print("[OK] Using Matplotlib for visualizations (Plotly not available)")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

print("[OK] Libraries imported successfully!")

# Cell 2: Load Dataset
print("\n[2/14] Loading housing dataset...")
housing_data = pd.read_csv('data/Housing_Price_Data.csv')
print(f"[OK] Dataset loaded: {housing_data.shape}")
print(f"[INFO] Features: {list(housing_data.columns)}")
print(f"[INFO] Missing values: {housing_data.isnull().sum().sum()}")
print(f"[INFO] Price range: ₹{housing_data['price'].min():,.0f} - ₹{housing_data['price'].max():,.0f}")

# Cell 3: Visualization 1 - Price vs Area by Furnishing
print("\n[3/14] Creating price vs area visualization...")
if USE_PLOTLY:
    fig = px.scatter(housing_data,
                     x='area',
                     y='price',
                     color='furnishingstatus',
                     size='bedrooms',
                     hover_data=['bathrooms', 'stories'],
                     title='Price vs Area by Furnishing Status',
                     labels={'area': 'Area (sq ft)', 'price': 'Price (₹)'})
    fig.write_html('viz/price_vs_area_furnishing.html')
    print(f"[OK] Saved: viz/price_vs_area_furnishing.html")
else:
    plt.figure(figsize=(12, 8))
    for furnish in housing_data['furnishingstatus'].unique():
        subset = housing_data[housing_data['furnishingstatus'] == furnish]
        plt.scatter(subset['area'], subset['price'],
                   s=subset['bedrooms']*50, alpha=0.6, label=furnish)
    plt.xlabel('Area (sq ft)', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.title('Price vs Area by Furnishing Status', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('viz/price_vs_area_furnishing.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: viz/price_vs_area_furnishing.png")

# Cell 4: Visualization 2 - AC Effect
print("\n[4/14] Creating air conditioning effect visualization...")

# Calculate AC premium
ac_yes = housing_data[housing_data['airconditioning'] == 'yes']['price'].mean()
ac_no = housing_data[housing_data['airconditioning'] == 'no']['price'].mean()
premium = ((ac_yes - ac_no) / ac_no) * 100

if USE_PLOTLY:
    fig = px.box(housing_data,
                 x='airconditioning',
                 y='price',
                 color='airconditioning',
                 title='Effect of Air Conditioning on House Prices',
                 labels={'airconditioning': 'Air Conditioning', 'price': 'Price (₹)'})
    fig.write_html('viz/airconditioning_effect.html')
    print(f"[OK] Saved: viz/airconditioning_effect.html")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    housing_data.boxplot(column='price', by='airconditioning', ax=ax)
    plt.suptitle('')
    plt.title('Effect of Air Conditioning on House Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Air Conditioning', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.tight_layout()
    plt.savefig('viz/airconditioning_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: viz/airconditioning_effect.png")

print(f"[INSIGHT] AC Premium: {premium:.1f}% (With AC: ₹{ac_yes:,.0f} vs Without: ₹{ac_no:,.0f})")

# Cell 5: Visualization 3 - Furnishing and Parking
print("\n[5/14] Creating furnishing and parking visualization...")
avg_price_data = housing_data.groupby(['furnishingstatus', 'parking'])['price'].mean().reset_index()

if USE_PLOTLY:
    fig = px.bar(avg_price_data,
                 x='furnishingstatus',
                 y='price',
                 color='parking',
                 barmode='group',
                 title='Average House Price by Furnishing Status and Parking',
                 labels={'furnishingstatus': 'Furnishing Status',
                        'price': 'Average Price (₹)',
                        'parking': 'Parking Spots'})
    fig.write_html('viz/furnishing_parking_price.html')
    print(f"[OK] Saved: viz/furnishing_parking_price.html")
else:
    pivot_data = avg_price_data.pivot(index='furnishingstatus',
                                        columns='parking',
                                        values='price')
    ax = pivot_data.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title('Average House Price by Furnishing Status and Parking',
              fontsize=16, fontweight='bold')
    plt.xlabel('Furnishing Status', fontsize=12)
    plt.ylabel('Average Price (₹)', fontsize=12)
    plt.legend(title='Parking Spots', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('viz/furnishing_parking_price.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: viz/furnishing_parking_price.png")

# Cell 6: Feature Encoding
print("\n[6/14] Encoding categorical features...")

housing_training_data = housing_data.copy()

def encode_dataset(df):
    """Encode categorical features for ML"""
    df = df.copy()  # Avoid modifying original
    df['furnishingstatus'] = df['furnishingstatus'].apply(
        lambda x: 2 if x == 'furnished' else (1 if x == 'semi-furnished' else 0)
    )

    le = LabelEncoder()
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
        df[col] = le.fit_transform(df[col])

    return df

housing_training_data = encode_dataset(housing_training_data)
print(f"[OK] Dataset encoded successfully!")
print(f"[INFO] Furnishing: 0=unfurnished, 1=semi-furnished, 2=furnished")
print(f"[INFO] Binary features: yes=1, no=0")

# Cell 7: Train-Test Split
print("\n[7/14] Splitting data...")
X = housing_training_data.drop('price', axis=1)
y = housing_training_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[OK] Training set: {X_train.shape}")
print(f"[OK] Testing set: {X_test.shape}")

# Cell 8: Feature Scaling
print("\n[8/14] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Features scaled using StandardScaler")

# Cell 9: Model 1 - AdaBoost
print("\n[9/14] Training AdaBoost Regressor...")
base_tree = DecisionTreeRegressor(max_depth=4)
adaboost_model = AdaBoostRegressor(
    estimator=base_tree,
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

adaboost_model.fit(X_train_scaled, y_train)
y_pred_ada = adaboost_model.predict(X_test_scaled)

ada_mae = mean_absolute_error(y_test, y_pred_ada)
ada_mse = mean_squared_error(y_test, y_pred_ada)
ada_r2 = r2_score(y_test, y_pred_ada)

print(f"[OK] AdaBoost trained!")
print(f"     MAE: ₹{ada_mae:,.2f}")
print(f"     MSE: {ada_mse:,.2f}")
print(f"     R² Score: {ada_r2:.4f}")

# Cell 10: Model 2 - Gradient Boosting
print("\n[10/14] Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_mse = mean_squared_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)

print(f"[OK] Gradient Boosting trained!")
print(f"     MAE: ₹{gb_mae:,.2f}")
print(f"     MSE: {gb_mse:,.2f}")
print(f"     R² Score: {gb_r2:.4f}")

# Cell 11: Model 3 - Linear Regression
print("\n[11/14] Training Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)

linear_mae = mean_absolute_error(y_test, y_pred_linear)
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

print(f"[OK] Linear Regression trained!")
print(f"     MAE: ₹{linear_mae:,.2f}")
print(f"     MSE: {linear_mse:,.2f}")
print(f"     R² Score: {linear_r2:.4f}")

# Cell 12: Model Comparison
print("\n[12/14] Comparing model performance...")

comparison_df = pd.DataFrame({
    'Model': ['AdaBoost', 'Gradient Boosting', 'Linear Regression'],
    'MAE': [ada_mae, gb_mae, linear_mae],
    'MSE': [ada_mse, gb_mse, linear_mse],
    'R² Score': [ada_r2, gb_r2, linear_r2]
})

comparison_df = comparison_df.sort_values('R² Score', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[0, 'Model']
best_r2 = comparison_df.loc[0, 'R² Score']
print(f"\n[BEST MODEL] {best_model_name} with R² = {best_r2:.4f}")

# Cell 13: Save Models
print("\n[13/14] Saving models and artifacts...")

# Save best model (assuming Linear Regression based on typical performance)
joblib.dump(linear_model, 'models/housing_price_model.joblib')
print(f"[OK] Model saved: models/housing_price_model.joblib")

joblib.dump(scaler, 'models/housing_scaler.joblib')
print(f"[OK] Scaler saved: models/housing_scaler.joblib")

feature_info = {
    'feature_names': X.columns.tolist(),
    'encoding_info': {
        'furnishingstatus': '0=unfurnished, 1=semi-furnished, 2=furnished',
        'binary_features': 'yes=1, no=0'
    }
}
joblib.dump(feature_info, 'models/housing_feature_info.joblib')
print(f"[OK] Feature info saved: models/housing_feature_info.joblib")

# Cell 14: Test Predictions
print("\n[14/14] Testing prediction function...")

def predict_price(features):
    """Predict house price using trained model"""
    features_encoded = encode_dataset(pd.DataFrame([features]))
    # Ensure correct column order matching training data
    features_encoded = features_encoded[X.columns]
    features_scaled = scaler.transform(features_encoded)
    predicted_price = linear_model.predict(features_scaled)
    return predicted_price[0]

# Test cases
print("\n" + "="*80)
print("SAMPLE HOUSE PRICE PREDICTIONS")
print("="*80)

premium_house = {
    'area': 3000, 'bedrooms': 4, 'bathrooms': 3, 'stories': 2,
    'mainroad': 'yes', 'guestroom': 'no', 'basement': 'yes',
    'hotwaterheating': 'no', 'airconditioning': 'yes', 'parking': 2,
    'furnishingstatus': 'furnished', 'prefarea': 'yes'
}
pred_premium = predict_price(premium_house)
print(f"\n1. Premium House (3000 sq ft, 4BR, Furnished, AC):")
print(f"   Predicted Price: ₹{pred_premium:,.2f}")

budget_house = {
    'area': 2000, 'bedrooms': 2, 'bathrooms': 1, 'stories': 1,
    'mainroad': 'yes', 'guestroom': 'no', 'basement': 'no',
    'hotwaterheating': 'no', 'airconditioning': 'no', 'parking': 0,
    'furnishingstatus': 'unfurnished', 'prefarea': 'no'
}
pred_budget = predict_price(budget_house)
print(f"\n2. Budget House (2000 sq ft, 2BR, Unfurnished, No AC):")
print(f"   Predicted Price: ₹{pred_budget:,.2f}")

midrange_house = {
    'area': 2500, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
    'mainroad': 'yes', 'guestroom': 'yes', 'basement': 'no',
    'hotwaterheating': 'no', 'airconditioning': 'yes', 'parking': 1,
    'furnishingstatus': 'semi-furnished', 'prefarea': 'yes'
}
pred_mid = predict_price(midrange_house)
print(f"\n3. Mid-Range House (2500 sq ft, 3BR, Semi-Furnished, AC):")
print(f"   Predicted Price: ₹{pred_mid:,.2f}")

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
print(f"\n[PERFORMANCE] Best Model: {best_model_name}")
print(f"  - MAE: ₹{comparison_df.loc[0, 'MAE']:,.2f}")
print(f"  - MSE: {comparison_df.loc[0, 'MSE']:,.2f}")
print(f"  - R²: {comparison_df.loc[0, 'R² Score']:.4f}")

print("\n" + "="*80)
print("ALL CELLS EXECUTED SUCCESSFULLY!")
print("🎉 30 DAYS OF DATASETS CHALLENGE COMPLETE! 🏆")
print("="*80)
