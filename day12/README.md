# Day 12: Wine Quality Classification

## Overview
Classification analysis of red wine quality based on physicochemical properties. Using machine learning to predict whether a wine is "Good" (quality ≥ 6) or "Not Good" (quality < 6) based on 11 chemical attributes.

## Dataset

**File:** `winequality-red.csv` (1,599 red wine samples)

**Source:** Wine Quality Dataset (UCI Machine Learning Repository)

**Features (11 physicochemical properties):**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

**Target:** Quality score (3-8), converted to binary classification (Good ≥ 6 vs Not Good < 6)

## Objectives

1. Explore wine quality distribution and feature characteristics
2. Analyze correlations between chemical properties and quality
3. Build binary classification models to predict wine quality
4. Identify most important features for quality prediction
5. Compare Logistic Regression vs Random Forest performance

## Analysis Performed

- Exploratory Data Analysis with 6 visualizations
- Feature distribution analysis by quality class
- Correlation analysis of physicochemical properties
- Binary classification modeling
- Feature importance ranking

## Models

**Model 1: Logistic Regression**
- Algorithm: L2-regularized logistic regression
- Features: 11 physicochemical properties (scaled)
- Target: Binary quality (0=Not Good, 1=Good)
- Performance: ~75-77% accuracy

**Model 2: Random Forest Classifier**
- Algorithm: Random Forest (100 trees, max_depth=10)
- Features: 11 physicochemical properties (unscaled)
- Target: Binary quality
- Performance: ~80-82% accuracy (best model)

## Key Findings

- **Quality Distribution**: Most wines rated 5-6, with few rated 3, 4, 7, or 8
- **Binary Split**: Approximately 53% "Not Good" (<6) vs 47% "Good" (≥6)
- **Top Predictors** (Random Forest feature importance):
  - Alcohol content (most important)
  - Volatile acidity
  - Sulphates
  - Citric acid
- **Alcohol Correlation**: Higher alcohol content tends to correlate with better quality
- **Acidity Balance**: Lower volatile acidity associated with higher quality
- **Model Performance**: Random Forest outperforms Logistic Regression by ~5%

## Files

```
day12/
├── data/
│   └── winequality-red.csv          # Red wine dataset (1,599 samples)
├── notebooks/
│   └── wine_quality_analysis.py     # Complete analysis script
├── models/
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   └── scaler.joblib
├── viz/
│   ├── quality_distribution.png
│   ├── feature_boxplots.png
│   ├── correlation_heatmap.png
│   ├── alcohol_vs_acidity_scatter.png
│   ├── model_performance.png
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

Run the analysis script:

```bash
cd day12/notebooks
python wine_quality_analysis.py
```

This will:
1. Load and explore the wine dataset
2. Generate 6 visualization files
3. Train Logistic Regression and Random Forest classifiers
4. Evaluate and compare model performance
5. Save all model artifacts

## Potential Extensions

- Multi-class classification (predict exact quality score 3-8)
- Include white wine dataset for comparison
- Ensemble stacking of multiple classifiers
- Regression to predict continuous quality score
- Feature engineering (ratios, interactions)
- Cross-validation for robust performance estimation

## Author
**Olayinka Akanji**
