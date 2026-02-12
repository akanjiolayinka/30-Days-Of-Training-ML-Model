# Day 15: Medical Insurance Cost Prediction

## Overview
Regression analysis to predict medical insurance charges based on demographic and health factors. Build and compare multiple regression models to understand which factors most significantly impact insurance costs.

## Dataset

**File:** `insurance.csv` (1,338 samples)

**Source:** Medical Cost Personal Datasets

**Features (6 predictors):**
- **age:** Age of primary beneficiary
- **sex:** Gender (male/female)
- **bmi:** Body Mass Index (kg/m²)
- **children:** Number of dependents covered
- **smoker:** Smoking status (yes/no)
- **region:** Residential area in the US (northeast, southeast, southwest, northwest)

**Target:** `charges` - Individual medical costs billed by health insurance ($)

## Objectives

1. Explore insurance cost distribution and patterns
2. Analyze impact of categorical variables (sex, smoker, region) on charges
3. Investigate relationships between continuous features (age, BMI) and costs
4. Build regression models to predict insurance charges
5. Compare Linear Regression, Ridge Regression, and Random Forest performance
6. Identify most important cost drivers

## Analysis Performed

- Exploratory Data Analysis with 8 visualizations
- Distribution analysis (raw and log-transformed charges)
- Categorical feature impact analysis (box plots)
- Continuous feature scatter plots with smoker stratification
- Correlation analysis of all features
- Regression modeling with 3 algorithms
- Feature importance ranking

## Models

**Model 1: Linear Regression**
- Algorithm: Ordinary Least Squares regression
- Features: 6 encoded features (scaled)
- Performance: R² ~0.75, RMSE ~$6,000

**Model 2: Ridge Regression**
- Algorithm: L2-regularized linear regression (alpha=1.0)
- Features: 6 encoded features (scaled)
- Performance: R² ~0.75, RMSE ~$6,000

**Model 3: Random Forest Regressor (Best Model)**
- Algorithm: Random Forest (100 trees, max_depth=15)
- Features: 6 encoded features (unscaled)
- Performance: R² ~0.86, RMSE ~$4,700, MAE ~$2,500

## Key Findings

- **Smoker Status:** The single most important predictor - smokers pay significantly higher insurance costs (~3-4x non-smokers)
- **Age Impact:** Older individuals generally have higher insurance costs, with substantial variation
- **BMI Correlation:** Higher BMI associated with increased costs, especially for smokers
- **Children:** Minimal impact on insurance costs
- **Regional Variation:** Small differences across regions (not a major driver)
- **Sex:** Minimal impact on costs
- **Distribution:** Insurance charges are right-skewed with a heavy tail (some very high-cost individuals)
- **Model Performance:** Random Forest significantly outperforms linear models (R² 0.86 vs 0.75)

## Files

```
day15/
├── data/
│   └── insurance.csv                    # Insurance dataset (1,338 samples)
├── notebooks/
│   └── insurance_cost_prediction.py     # Complete analysis script
├── models/
│   ├── linear_regression_model.joblib
│   ├── ridge_regression_model.joblib
│   ├── random_forest_model.joblib       # Best model
│   ├── scaler.joblib
│   ├── sex_encoder.joblib
│   ├── smoker_encoder.joblib
│   └── region_encoder.joblib
├── viz/
│   ├── charges_distribution.png
│   ├── charges_by_categories.png
│   ├── continuous_features_scatter.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importance.png
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

## Usage

Run the insurance cost prediction analysis:

```bash
cd day15/notebooks
python insurance_cost_prediction.py
```

This will:
1. Load and explore the insurance dataset
2. Encode categorical variables
3. Generate 8 visualization files
4. Train 3 regression models (Linear, Ridge, Random Forest)
5. Compare model performance
6. Save all 7 model artifacts to models/

## Potential Extensions

- Polynomial feature engineering (age², BMI², age×BMI, age×smoker)
- Interaction terms to capture non-linear relationships
- Gradient Boosting or XGBoost for improved performance
- Geographic analysis if zip code data available
- Cost category classification (Low/Medium/High)
- Outlier analysis for extreme-cost individuals
- Cross-validation for robust performance estimation

## Author
**Olayinka Akanji**
