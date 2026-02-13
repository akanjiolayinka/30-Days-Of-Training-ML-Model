# Day 30: Housing Price Prediction - Final Project 🏡

## Dataset
**Housing Price Data**

## Project Overview
**🎉 FINAL DAY OF 30 DAYS OF DATASETS! 🎉**

This capstone project analyzes housing prices to identify key market drivers and build predictive models for real estate valuation. We explore relationships between property features (area, bedrooms, amenities) and pricing, then compare three regression models to determine the best performer.

## Dataset Details
- **File**: Housing_Price_Data.csv
- **Features**: 12 property attributes
- **Target**: House price

### Features
**Property Characteristics:**
- `area` - Property size (square feet)
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `stories` - Number of floors
- `parking` - Number of parking spaces

**Amenities (Binary):**
- `mainroad` - Access to main road (yes/no)
- `guestroom` - Presence of guest room (yes/no)
- `basement` - Basement availability (yes/no)
- `hotwaterheating` - Hot water heating system (yes/no)
- `airconditioning` - AC availability (yes/no)
- `prefarea` - Located in preferred area (yes/no)

**Furnishing Status (Categorical):**
- `furnishingstatus` - furnished, semi-furnished, unfurnished

## Objectives
1. Identify key price drivers through exploratory data analysis
2. Visualize market dynamics (area vs price, AC impact, furnishing & parking)
3. Compare regression models (AdaBoost, Gradient Boosting, Linear Regression)
4. Build production-ready price prediction system

## Key Findings

### Market Drivers Priority

| Factor | Impact | Priority |
|--------|--------|----------|
| Area (Size) | Strongest predictor - direct positive correlation | Critical |
| Air Conditioning | ~₹2M price premium (50% premium) | High |
| Furnishing Status | Furnished adds 30-40% premium vs unfurnished | High |
| Parking Availability | 3+ spots add significant premium | Medium-High |
| Bedrooms | Correlates with both area and price | Medium |

### Price Insights
- **Price Range**: ₹2M to ₹14M
- **Area Range**: 1,500 to 16,500 square units
- **Market Concentration**: Most properties cluster between 4,000-8,000 sq units at ₹4M-7M
- **AC Premium**: Properties with AC command ~₹2M higher prices (50% increase)
- **Furnishing Impact**:
  - Furnished: ~₹25M total average value
  - Semi-Furnished: ~₹20M total average value
  - Unfurnished: ~₹18M total average value

## Visualizations

All visualizations saved to `viz/` directory as interactive HTML files:

### 1. Price vs Area by Furnishing Status (`price_vs_area_furnishing.html`)
**Bubble Scatter Plot**

**Key Observations:**
- Strong positive correlation between area and price across all furnishing categories
- Furnished (Blue): Mid-to-high price range with premium positioning
- Semi-Furnished (Red): Mid-range pricing with moderate spread
- Unfurnished (Green): Widest distribution, generally lower prices for same area
- Bubble size (bedroom count) shows larger homes tend to be furnished

### 2. Effect of Air Conditioning on Prices (`airconditioning_effect.html`)
**Box Plot Comparison**

**Key Observations:**
- **With AC**: Median ₹6M, range ₹2M-₹10M+ (outliers to ₹13M)
- **Without AC**: Median ₹4M, range ₹1.5M-₹8M
- **Price Difference**: ~₹2M premium (50% increase) for AC-equipped properties
- AC is a significant value-added feature in the real estate market

### 3. Furnishing Status and Parking Impact (`furnishing_parking_price.html`)
**Grouped Bar Chart**

**Market Segments:**
- **Furnished + High Parking**: Premium segment (~₹10M-12M)
- **Semi-Furnished + Medium Parking**: Mid-market segment (~₹8M-10M)
- **Unfurnished + Low Parking**: Entry-level segment (~₹4M-6M)

**Parking Impact:**
- 3+ parking spots command highest premium
- 1-2 spots add mid-level value
- 0-1 spots represent base value

## Model Performance

### Models Compared

We trained and evaluated three regression models:

1. **AdaBoost Regressor**
   - Base Estimator: Decision Tree (max_depth=4)
   - N_estimators: 200
   - Learning Rate: 0.05

2. **Gradient Boosting Regressor**
   - N_estimators: 300
   - Learning Rate: 0.05
   - Max Depth: 4

3. **Linear Regression**
   - Baseline model
   - Simple and interpretable

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

### Best Performing Model
The model comparison identifies the best performer based on R² score, with all models providing production-ready predictions for real estate valuation.

## Feature Engineering

### Encoding Strategy
1. **Furnishing Status**: Ordinal encoding
   - 0 = unfurnished
   - 1 = semi-furnished
   - 2 = furnished

2. **Binary Features**: Label encoding (yes=1, no=0)
   - mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea

3. **Feature Scaling**: StandardScaler normalization for all features

## Market Recommendations

### For Buyers
- **Area is the primary value driver** - focus on location and size
- AC presence is highly valued - consider maintenance costs
- Parking availability should influence location choice
- Unfurnished properties offer better value for customization

### For Sellers
- **Improving furnishing status** can increase appeal by 30-40%
- **Adding parking facilities** provides good ROI
- **Air conditioning upgrade** justified by 50% price premium
- Properties near main roads command higher prices

### For Investors
- **Unfurnished properties in high-demand areas** offer best ROI potential
- Focus on properties with expansion/renovation opportunities
- Market shows strong demand for both luxury and value segments
- **AC and parking upgrades** provide measurable value addition

## Files Structure

```
day30/
├── data/
│   └── Housing_Price_Data.csv           # Original dataset
├── notebooks/
│   └── housing_price_prediction.ipynb   # Complete analysis & ML pipeline
├── viz/
│   ├── price_vs_area_furnishing.html   # Scatter plot
│   ├── airconditioning_effect.html     # Box plot
│   └── furnishing_parking_price.html   # Bar chart
├── models/
│   ├── housing_price_model.joblib      # Trained model
│   ├── housing_scaler.joblib           # StandardScaler
│   └── housing_feature_info.joblib     # Feature metadata
├── summary/
│   └── housing_price_summary.md        # Technical summary
└── README.md
```

## Technologies Used
- **Python 3.x**
- **pandas & numpy** - Data manipulation
- **plotly** - Interactive visualizations
- **scikit-learn** - Machine learning
  - AdaBoostRegressor, GradientBoostingRegressor, LinearRegression
  - StandardScaler, LabelEncoder
  - train_test_split, regression metrics
- **joblib** - Model persistence

## How to Run

### Prerequisites
```bash
pip install pandas numpy plotly scikit-learn joblib
```

### Execution
1. Navigate to `notebooks/` directory
2. Open `housing_price_prediction.ipynb`
3. Run all cells to:
   - Load and explore housing data
   - Generate 3 interactive visualizations
   - Train 3 regression models
   - Compare model performance
   - Save best model for deployment

### Using the Prediction Function
```python
# Example: Predict price for a new house
new_house = {
    'area': 3000,
    'bedrooms': 4,
    'bathrooms': 3,
    'stories': 2,
    'mainroad': 'yes',
    'guestroom': 'no',
    'basement': 'yes',
    'hotwaterheating': 'no',
    'airconditioning': 'yes',
    'parking': 2,
    'furnishingstatus': 'furnished',
    'prefarea': 'yes'
}

predicted_price = predict_price(new_house)
print(f'Predicted Price: ₹{predicted_price:,.2f}')
```

## Key Takeaways

### Data Insights
1. **Area is King**: Property size remains the strongest predictor of housing prices
2. **Amenities Matter**: AC, parking, and furnishing significantly impact valuation
3. **Premium Justification**: Higher-end features command measurable price premiums
4. **Market Segmentation**: Clear price tiers exist based on feature combinations

### Model Insights
1. **Ensemble Methods**: AdaBoost and Gradient Boosting provide robust predictions
2. **Linear Baseline**: Simple linear regression serves as effective baseline
3. **Predictive Accuracy**: Models provide reliable price predictions for real estate valuation
4. **Production Ready**: Saved models, scalers, and prediction functions for deployment

### Business Value
- **Automated price estimation** reduces manual effort
- **Data-driven pricing** improves profitability
- **Consumer transparency** builds trust
- **Strategic improvements** (AC, parking, furnishing) offer quantifiable ROI
- **Investment strategies**: ROI opportunities through targeted upgrades

## Conclusion

**🎉 30 Days of Datasets Complete! 🎉**

This final project demonstrates a complete data science workflow:
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data visualization and insights extraction
- ✅ Feature engineering and preprocessing
- ✅ Multiple model comparison
- ✅ Production deployment preparation
- ✅ Business recommendations

**Model Applications:**
- Real estate price estimation
- Market analysis and segmentation
- Investment opportunity identification
- Property valuation for buyers/sellers
- Automated appraisal systems

---
*Day 30 - Final Project Complete!*
*30 Days of Datasets Challenge Successfully Finished! 🏆*
