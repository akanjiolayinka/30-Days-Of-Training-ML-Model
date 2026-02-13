# Day 7: Fruit Classification - ML Classification Models

## Dataset
**Fruit Classification Dataset**

Source: Kaggle

## Project Overview
This project provides comprehensive exploratory data analysis (EDA) and machine learning classification for a fruit dataset containing 10,000 samples across 20 different fruit types. Using physical characteristics (size, weight, price) and sensory attributes (color, shape, taste), we build a Random Forest classifier achieving perfect or near-perfect classification accuracy.

The analysis includes interactive Plotly visualizations, feature engineering, and a complete ML pipeline from data exploration to model deployment.

## Dataset Details
- **Size**: 10,000 samples
- **Features**: 7 columns (3 numerical, 3 categorical, 1 target)
- **Target**: fruit_name (20 unique fruit types)

### Features
**Numerical:**
- `size (cm)` - Physical size in centimeters
- `weight (g)` - Weight in grams
- `avg_price (₹)` - Average price in Indian Rupees

**Categorical:**
- `shape` - Fruit shape (long, oval, round)
- `color` - Fruit color (multiple colors)
- `taste` - Taste profile (sweet, sour, tangy)

**Target:**
- `fruit_name` - 20 fruit types (apple, banana, blueberry, cherry, coconut, custard apple, dragon fruit, grape, guava, kiwi, lychee, mango, orange, papaya, pear, pineapple, plum, pomegranate, strawberry, watermelon)

## Obiective
Build a classification model to predict fruit type based on physical and sensory characteristics, exploring relationships between features and identifying the most important predictors.

## Exploratory Data Analysis

### Data Quality
- **Missing Values**: 0 (complete dataset)
- **Class Balance**: Well-distributed (~400-550 samples per fruit)
- **Feature Range**:
  - Size: 0.9 - 27.5 cm
  - Weight: 4.5 - 3299.8 g
  - Price: ₹9 - ₹165

### Key Insights
1. **Size-Weight Correlation**: Strong positive correlation (0.94) between size and weight
2. **Price Patterns**: Larger fruits (watermelon, pineapple) command higher prices
3. **Color Distribution**: Green fruits dominate the dataset
4. **Taste Profile**: Sweet taste is most common across all fruits
5. **Shape Diversity**: Round fruits are most prevalent

## Feature Engineering

### Encoding Strategy
1. **Target Variable**: Label encoding for fruit_name (0-19)
2. **Categorical Features**: One-hot encoding with drop_first=True
   - `shape` → shape_oval, shape_round
   - `color` → 7 color dummies
   - `taste` → taste_sweet, taste_tangy

### Final Feature Set
- **Total Features**: 14 after encoding
- **Numerical**: 3 (size, weight, price)
- **Encoded Categorical**: 11 (shapes, colors, tastes)

## Model Performance

### Random Forest Classifier
**Configuration:**
- Estimators: 100
- Random State: 42
- n_jobs: -1 (parallel processing)

**Results:**
- **Test Accuracy**: ~100% (perfect classification)
- **Training Accuracy**: ~100%
- **Precision**: 1.00 (all classes)
- **Recall**: 1.00 (all classes)
- **F1-Score**: 1.00 (all classes)

### Feature Importance Analysis

Top predictive features:
1. **weight (g)** - ~18.4% importance
2. **avg_price (₹)** - ~16.9% importance
3. **size (cm)** - ~13.1% importance
4. **color_green**, **color_orange**, **shape_oval** - Moderate importance
5. **taste_sweet**, **taste_tangy** - Lower importance

**Key Finding**: Physical attributes (size, weight, price) are the strongest predictors, accounting for ~48% of model decisions, while sensory attributes provide additional discriminative power.

## Visualizations

All visualizations are saved to the `viz/` directory as interactive HTML files:

### 1. Numerical Distributions (`numerical_distributions.html`)
- Histograms of size, weight, and price distributions
- Shows range and spread of physical characteristics

### 2. Feature-by-Fruit Distributions
- `size_cm_by_fruit.html` - Size distribution colored by fruit type
- `weight_g_by_fruit.html` - Weight distribution colored by fruit type
- `avg_price_inr_by_fruit.html` - Price distribution colored by fruit type
- Reveals how each fruit type clusters in feature space

### 3. Correlation Heatmap (`correlation_heatmap.html`)
- Shows strong size-weight correlation (0.94)
- Identifies price-weight relationship (0.68)
- Interactive correlation matrix for numerical features

### 4. Average Price by Fruit (`avg_price_by_fruit.html`)
- Bar chart showing price hierarchy across fruits
- Watermelon, dragon fruit, and pineapple are most expensive
- Berries and small fruits are more affordable

### 5. Size vs Weight Scatter (`size_vs_weight_scatter.html`)
- Bubble plot with price as bubble size
- Color-coded by fruit type
- Hover shows additional attributes (color, taste)
- Clearly shows fruit clustering patterns

### 6. Taste by Color Distribution (`taste_by_color.html`)
- Grouped histogram of taste profiles by color
- Reveals color-taste associations
- Interactive exploration of categorical relationships

### 7. Feature Importance (`feature_importance.html`)
- Top 10 features ranked by Random Forest importance
- Horizontal bar chart for easy comparison
- Helps identify key discriminators

### 8. Confusion Matrix (`confusion_matrix.html`)
- Annotated heatmap showing prediction accuracy
- Perfect diagonal (all correct predictions)
- 20x20 matrix for all fruit classes
- Interactive tooltips with actual values

## Files Structure

```
day7/
├── data/
│   └── fruit_classification_dataset.csv  # Original dataset (10,000 samples)
├── notebooks/
│   └── fruit_classification.ipynb        # Complete EDA & ML pipeline
├── viz/
│   ├── numerical_distributions.html      # Feature distributions
│   ├── size_cm_by_fruit.html            # Size by fruit type
│   ├── weight_g_by_fruit.html           # Weight by fruit type
│   ├── avg_price_inr_by_fruit.html      # Price by fruit type
│   ├── correlation_heatmap.html          # Feature correlations
│   ├── avg_price_by_fruit.html          # Price comparison
│   ├── size_vs_weight_scatter.html      # Interactive scatter plot
│   ├── taste_by_color.html              # Categorical analysis
│   ├── feature_importance.html           # Feature rankings
│   └── confusion_matrix.html             # Model performance
├── models/
│   ├── fruit_rf_model.joblib            # Trained Random Forest
│   └── label_encoder.joblib             # Label encoder for fruits
├── summary/
└── README.md
```

## Technologies Used
- **Python 3.x**
- **pandas & numpy** - Data manipulation
- **scikit-learn** - Machine learning
  - RandomForestClassifier
  - LabelEncoder
  - train_test_split
  - Classification metrics
- **Plotly** - Interactive visualizations
  - plotly.express
  - plotly.graph_objects
  - plotly.figure_factory
- **joblib** - Model persistence

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn plotly joblib
```

### Execution
1. Navigate to `notebooks/` directory
2. Open `fruit_classification.ipynb` in Jupyter Notebook
3. Run all cells sequentially to:
   - Load and explore the dataset
   - Create interactive visualizations (saved to viz/)
   - Encode categorical features
   - Train Random Forest classifier
   - Evaluate model performance
   - Save trained model and visualizations

### Test Script
```bash
cd day7
python test_pipeline.py
```
- Tests core ML pipeline without Plotly
- Validates data loading and model training
- Verifies model performance

## Model Deployment

### Loading Saved Model
```python
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load('models/fruit_rf_model.joblib')
encoder = joblib.load('models/label_encoder.joblib')

# Predict new fruit
sample = pd.DataFrame([{
    'size (cm)': 25.0,
    'weight (g)': 3000.0,
    'avg_price (₹)': 140.0,
    # ... encoded categorical features
}])

prediction = model.predict(sample)
fruit_name = encoder.inverse_transform(prediction)[0]
print(f"Predicted fruit: {fruit_name}")
```

## Key Results

### Classification Performance
- **Perfect Accuracy**: 100% on test set
- **No Overfitting**: Training and test accuracy aligned
- **All Classes**: 100% precision, recall, and F1-score
- **Robust Model**: Stratified sampling ensures balanced evaluation

### Feature Insights
1. **Weight dominates**: Single strongest predictor (18.4%)
2. **Price follows weight**: Second most important (16.9%)
3. **Size complements**: Third predictor (13.1%)
4. **Combined physical features**: Account for 48.4% of decisions
5. **Categorical features**: Provide supplementary discrimination (51.6%)

### Business Value
- **Automated fruit identification** from physical measurements
- **Quality control** in fruit processing
- **Pricing optimization** based on fruit characteristics
- **Inventory management** using automated classification
- **Market analysis** of fruit attributes and pricing

## Conclusion

The Random Forest model achieves exceptional performance (100% accuracy) on the fruit classification task, demonstrating that physical characteristics (size, weight, price) combined with sensory attributes (shape, color, taste) provide sufficient information for perfect fruit discrimination.

**Key Takeaways**:
- Physical measurements are highly predictive
- Dataset has well-separated fruit classes
- No additional feature engineering needed
- Model ready for production deployment
- Interactive visualizations enable deep exploration

---
*Analysis completed as part of 30 Days of Datasets*
*Model achieves perfect classification (100% accuracy)*
