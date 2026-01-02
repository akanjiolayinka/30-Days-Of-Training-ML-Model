# Day 22: Housing Price Analysis & Prediction

## Project Overview

This project analyzes real estate market dynamics to understand housing price determinants and build accurate price prediction models. By examining property characteristics, location factors, and market conditions, we provide insights for buyers, sellers, and investors in the residential real estate market.

## Dataset

**File:** `Housing_Price_Data.csv`

**Description:** Comprehensive residential property listings with detailed features and sale prices

**Key Features:**
- **Property Characteristics:** Square footage, number of bedrooms/bathrooms, lot size
- **Building Details:** Year built, property age, construction quality
- **Amenities:** Air conditioning, parking spaces, furnishing status, swimming pool
- **Location:** Neighborhood, proximity to amenities, school district ratings
- **Market Data:** Sale price, listing price, days on market
- **Condition:** Renovation status, maintenance quality

## Objectives

1. **Market Analysis:**
   - Price distribution across different neighborhoods
   - Average price per square foot by location
   - Property type (apartment, house, villa) price comparison
   - Seasonal trends in pricing (if temporal data available)

2. **Feature Impact Assessment:**
   - Square footage correlation with price
   - Bedroom/bathroom count influence on value
   - Air conditioning premium quantification
   - Furnishing status impact on pricing
   - Parking availability value assessment
   - Age vs price relationship (depreciation curve)

3. **Predictive Modeling:**
   - Price prediction using regression algorithms:
     - Linear Regression (baseline)
     - Random Forest Regressor
     - Gradient Boosting (XGBoost, LightGBM)
     - AdaBoost Regressor
   - Feature importance analysis
   - Model evaluation (R², MAE, RMSE, MAPE)

4. **Investment Insights:**
   - Undervalued property identification
   - Price appreciation potential by neighborhood
   - ROI analysis for renovation investments
   - Market valuation benchmarks

## Analysis Techniques

- Regression modeling with cross-validation
- Feature engineering (price per sqft, age categories, amenity scores)
- Outlier detection and treatment
- Categorical encoding (one-hot, target encoding)
- Hyperparameter tuning with GridSearch
- Residual analysis for model diagnostics
- Geographic analysis and mapping

## Expected Outcomes

- Predictive model with R² > 0.85
- Square footage identified as strongest price predictor
- AC premium quantification (~50% value increase expected)
- Furnishing status impact analysis (30-40% premium)
- Parking space value contribution
- Neighborhood price tier classification
- Interactive price estimation tool
- Comprehensive market insights report

## Visualizations

- Price distribution histograms and KDE plots
- Scatter plots (Area vs Price, Age vs Price)
- Feature correlation heatmap
- Geographic price heatmaps by neighborhood
- Box plots for categorical feature comparisons
- Feature importance bar charts
- Actual vs Predicted price scatter plots
- Residual distribution plots
- Price per sqft trends

## Key Insights Expected

- Area (square footage) is the dominant price driver
- AC adds significant premium (~50%)
- Furnished properties command 30-40% higher prices
- Parking availability increases value by 15-25%
- Location/neighborhood has substantial impact
- Property age shows depreciation curve
- Amenity count positively correlates with price

## Project Structure

- `data/` - Housing price dataset with data dictionary
- `models/` - Trained regression models and price prediction pipelines
- `notebooks/` - EDA, feature engineering, and modeling notebooks
- `viz/` - Price analysis charts, market insights, prediction visualizations

## Getting Started

Start with the EDA notebook to understand price distributions and key market drivers, then explore the modeling notebook to see how different features contribute to accurate price predictions.
