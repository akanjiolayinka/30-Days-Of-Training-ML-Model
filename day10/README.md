# Day 10: Medical Insurance Cost Analysis & Prediction

## Project Overview

This project analyzes medical insurance costs and identifies the key factors driving healthcare expenses. Using demographic, lifestyle, and health-related features, we build predictive models to estimate insurance charges and provide insights for both insurers and policyholders on cost determinants.

## Dataset

**File:** `insurance.csv`

**Description:** Medical insurance records with personal attributes and associated charges

**Key Features:**
- **Demographic:** Age, gender, number of dependents (children)
- **Geographic:** Region of residence
- **Health Indicators:** BMI (Body Mass Index), smoking status
- **Target Variable:** Medical insurance charges (continuous)

## Objectives

1. **Exploratory Data Analysis:**
   - Distribution of insurance charges across population
   - Age, BMI, and charges relationship analysis
   - Smoking status impact on medical costs
   - Regional variations in healthcare expenses
   - Gender and family size effects on charges

2. **Cost Driver Identification:**
   - Statistical testing for significant cost factors
   - Correlation analysis between features and charges
   - Segment-wise cost profiling (smokers vs non-smokers, regions, age groups)
   - Outlier analysis for high-cost cases

3. **Predictive Modeling:**
   - Regression models to predict insurance charges:
     - Linear Regression (baseline)
     - Decision Tree Regressor
     - Random Forest Regressor
     - Gradient Boosting (XGBoost, LightGBM)
   - Feature importance ranking
   - Model performance comparison (R², MAE, RMSE)

4. **Business Insights:**
   - Risk profiling and premium recommendations
   - Cost reduction opportunities identification
   - High-risk segment characterization
   - Policy pricing insights

## Analysis Techniques

- Univariate and multivariate statistical analysis
- Feature encoding for categorical variables
- Feature engineering (age groups, BMI categories, interaction terms)
- Regression modeling with cross-validation
- Hyperparameter tuning for optimal performance
- Residual analysis and model diagnostics
- Feature importance interpretation

## Expected Outcomes

- Predictive model with R² > 0.75
- Clear identification of top cost drivers
- Smoking status impact quantification (~3-4x cost multiplier expected)
- BMI threshold analysis for cost escalation
- Regional cost variation mapping
- Interactive dashboards showing cost breakdowns
- Policy recommendations based on risk factors
- Prediction tool for individual charge estimation

## Visualizations

- Charge distribution histograms and KDE plots
- Scatter plots (Age vs Charges, BMI vs Charges) with smoking status colors
- Box plots comparing charges across categories
- Correlation heatmap of numerical features
- Feature importance bar charts
- Residual plots for model diagnostics
- Regional cost comparison maps
- Actual vs Predicted scatter plots

## Project Structure

- `data/` - Insurance dataset with data dictionary
- `models/` - Trained regression models and preprocessing pipelines
- `notebooks/` - EDA, feature engineering, and modeling notebooks
- `viz/` - Cost analysis plots, model performance charts, dashboards

## Getting Started

Explore the EDA notebook first to understand cost distributions and key drivers, then review the modeling notebook to see how different algorithms perform in predicting insurance charges.
