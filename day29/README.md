# Day 29: Car Price Prediction & Automotive Market Analysis

## Project Overview

This project analyzes the automotive market to understand vehicle pricing dynamics and build accurate price prediction models. By examining car specifications, brand reputation, mileage, age, and market conditions, we provide insights for buyers, sellers, and dealers in the used and new car market.

## Dataset

**File:** `car_price_prediction_.csv`

**Description:** Comprehensive vehicle listings with specifications and pricing data

**Key Features:**
- **Vehicle Details:** Make, model, trim level, year
- **Specifications:** Engine size, horsepower, transmission type, fuel type
- **Condition:** Mileage, age, previous owners
- **Features:** Sunroof, leather seats, navigation, safety features
- **Market Data:** Listing price, market value, days on market
- **Location:** Dealer location, regional market indicators

## Objectives

1. **Market Analysis:**
   - Price distribution across brands and models
   - Depreciation curves by vehicle age
   - Mileage impact on valuation
   - Brand premium quantification (luxury vs economy)

2. **Specification Impact:**
   - Engine size and horsepower correlation with price
   - Transmission type (automatic vs manual) value difference
   - Fuel type trends (petrol, diesel, hybrid, electric)
   - Feature package impact on pricing

3. **Depreciation Analysis:**
   - Year-over-year depreciation rates
   - Mileage per year optimal ranges
   - Brand-specific depreciation patterns
   - Electric/hybrid vs traditional depreciation comparison

4. **Predictive Modeling:**
   - Car price prediction using regression models:
     - Random Forest Regressor
     - Gradient Boosting (XGBoost, LightGBM)
     - AdaBoost Regressor
     - Stacked Ensemble
   - Feature importance analysis
   - Model evaluation (R², MAE, RMSE, MAPE)

## Analysis Techniques

- Regression modeling with cross-validation
- Feature engineering (age, mileage per year, brand encoding)
- Categorical encoding (one-hot, target encoding)
- Outlier detection and treatment (luxury/exotic cars)
- Hyperparameter tuning with GridSearch
- Time-based depreciation modeling
- Brand reputation scoring

## Expected Outcomes

- Predictive model with R² > 0.90
- Car age and mileage identified as top depreciation factors
- Luxury brand premium quantification (2-3x base models)
- Electric/hybrid price trends analysis
- Transmission type value difference (~5-10%)
- Feature importance rankings for valuation
- Interactive price estimation tool
- Depreciation calculator by brand and model

## Visualizations

- Price distribution histograms by brand
- Depreciation curves (price vs age)
- Scatter plots (Mileage vs Price, Age vs Price)
- Brand comparison box plots
- Feature correlation heatmap
- Feature importance bar charts
- Actual vs Predicted price scatter plots
- Regional price variation maps
- Fuel type popularity trends

## Key Insights Expected

- **Car age** is the strongest depreciation factor
- **Mileage per year** matters more than total mileage
- Luxury brands (BMW, Mercedes, Audi) retain value better
- Electric vehicles show different depreciation patterns
- Automatic transmission commands premium in most markets
- First 3 years see steepest depreciation (~40-50%)
- Feature packages add 10-20% to base value
- Diesel vehicles popular in specific markets

## Advanced Features

- **Depreciation Rate:** Calculate yearly value loss percentage
- **Mileage Per Year:** Total mileage / vehicle age
- **Brand Premium Score:** Luxury vs economy brand encoding
- **Feature Count:** Total number of premium features
- **Market Position:** Price percentile within brand/model

## Project Structure

- `data/` - Car pricing dataset with data dictionary
- `models/` - Trained regression models and price prediction pipelines
- `notebooks/` - EDA, depreciation analysis, and predictive modeling notebooks
- `viz/` - Price trends, brand comparisons, depreciation curves, market insights

## Getting Started

Begin with the EDA notebook to understand pricing distributions and depreciation patterns, then explore the modeling notebook to see how various features contribute to accurate car price predictions.
