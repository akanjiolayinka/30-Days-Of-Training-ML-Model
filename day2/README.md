# Day 2: Earthquake & Tsunami Risk Assessment

## Dataset
**Global Earthquake & Tsunami Risk Assessment Dataset**

Source: https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/data

## Project Overview
Built a machine learning model to predict tsunami occurrence based on earthquake characteristics. The Random Forest classifier achieved **80-90% accuracy** in identifying tsunami risk from seismic data.

This project analyzes global seismic activity data to understand earthquake patterns and predict tsunami occurrence risk using comprehensive feature engineering and machine learning techniques.

## Objective
Predict whether an earthquake will trigger a tsunami based on seismic features including:
- Magnitude and seismic energy
- Depth and geographic location
- Intensity measurements (CDI, MMI)
- Temporal patterns
- Proximity metrics

## Feature Engineering

Created additional features to improve model performance:

1. **Energy**: Magnitude squared to represent seismic energy release
   - Formula: `energy = magnitude²`
   - Captures exponential energy increase with magnitude

2. **Felt vs Measured**: Difference between MMI and CDI intensity scales
   - Formula: `felt_vs_measured = mmi - cdi`
   - Reveals perception vs measurement discrepancies

3. **Distance Metrics**: Converted dmin to kilometers and created proximity score
   - `dmin_km = dmin * 111` (degrees to km conversion)
   - `proximity_score = 1 / (dmin_km + 1)`
   - Higher scores indicate closer epicenters

4. **Gap Normalization**: Standardized azimuthal gap to 0-1 scale
   - Formula: `gap_norm = gap / 360`
   - Represents seismic station coverage

5. **Depth Categories**: Classified earthquakes as shallow (0-70km), intermediate (70-300km), or deep (300-700km)
   - Shallow earthquakes pose greater tsunami risk
   - One-hot encoded for model input

6. **Seasonal Patterns**: Encoded month into seasonal categories
   - Formula: `season = (Month % 12) // 3 + 1`
   - Explores potential temporal patterns

## Model Performance

### Random Forest Classifier

**Model Specifications:**
- Training samples: 80% of dataset (625 samples)
- Test samples: 20% of dataset (157 samples)
- Number of estimators: 300
- Feature scaling: StandardScaler applied
- Random state: 42 (reproducibility)

**Performance Metrics:**
- **Accuracy**: 80-90%
- **Precision**: ~90% (Class 0), ~83% (Class 1)
- **Recall**: ~86% (Class 0), ~95% (Class 1)
- **F1-Score**: Balanced performance across classes

The model successfully identifies patterns between earthquake characteristics and tsunami occurrence, with feature importance analysis revealing key predictive factors.

## Key Findings

### Most Important Features

1. **Year of occurrence** - Indicating climate change impact on seismic patterns
2. **Magnitude and derived energy** - Stronger quakes more likely to trigger tsunamis
3. **Depth and proximity metrics** - Shallow, nearby earthquakes pose greatest risk
4. **Geographic coordinates** - Certain regions more susceptible
5. **Intensity measurements** (MMI, CDI) - Ground shaking indicators

### Model Insights

- **Temporal trends** show increasing tsunami probability over time
- **Shallow earthquakes** with high magnitude pose greatest tsunami risk
- **Proximity to epicenter** and seismic intensity are strong indicators
- **Climate change connection**: Year as top feature suggests environmental factors

⚠️ **Important Finding**: The fact that "Year" of occurrence is the most important feature goes to show how global warming has affected the probability of tsunamis.

## Visualizations

### 1. Correlation Heatmap
![Correlation Heatmap](viz/correlation_heatmap.png)
- Shows relationships between all features
- Identifies multicollinearity
- Highlights tsunami correlations

### 2. Feature Importance
![Feature Importance](viz/feature_importance.png)
- Top 10 tsunami contributors ranked
- Reveals key predictive factors
- Guides feature selection

### 3. Confusion Matrix
![Confusion Matrix](viz/confusion_matrix.png)
- Model prediction accuracy breakdown
- True/False positive/negative rates
- Performance visualization

## Technical Details

### Preprocessing
- **Standard scaling** applied to numerical features
- **One-hot encoding** for categorical variables (depth category, season)
- **Train-test split** with random state for reproducibility
- **No missing values** in dataset

### Evaluation Metrics
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Feature importance ranking
- Probability predictions for risk assessment

### Model Testing

Included prediction example for hypothetical earthquake:
- **Magnitude**: 7.2
- **Depth**: 50km (shallow)
- **Location**: 38.322°N, 142.369°E (Near Japan)
- **Year**: 2024, Month: March
- **High seismic signature**
- **Result**: Model successfully classifies tsunami risk

## Files Structure

```
day2/
├── data/
│   └── earthquake_data_tsunami.csv  # Original dataset
├── notebooks/
│   └── earthquake_tsunami_ml.ipynb  # Complete ML pipeline
├── viz/
│   ├── correlation_heatmap.png      # Feature correlations
│   ├── feature_importance.png       # Top predictive features
│   └── confusion_matrix.png         # Model performance
├── models/
│   ├── tsunami_rf_model.joblib      # Trained Random Forest model
│   └── scaler.joblib                # StandardScaler for preprocessing
├── summary/
└── README.md
```

## Technologies Used

- **Python 3.x**
- **pandas & numpy** - Data manipulation
- **scikit-learn** - Machine learning
  - RandomForestClassifier
  - StandardScaler
  - train_test_split
  - classification metrics
- **seaborn & matplotlib** - Visualizations
- **joblib** - Model persistence

## How to Run

1. Navigate to `notebooks/` directory
2. Open `earthquake_tsunami_ml.ipynb` in Jupyter Notebook
3. Run all cells sequentially to:
   - Load and explore earthquake data
   - Engineer features
   - Train Random Forest model
   - Evaluate performance
   - Test on new scenarios
   - Save trained model

## Model Applications

- **Early warning systems** for coastal communities
- **Risk assessment** for disaster preparedness
- **Insurance modeling** for tsunami-prone regions
- **Urban planning** in seismic zones
- **Evacuation planning** based on risk scores

## Key Insights for Risk Assessment

### High-Risk Indicators
- Magnitude > 7.0
- Shallow depth (< 70km)
- Recent occurrence (temporal trends)
- High seismic energy (magnitude²)
- Coastal proximity

### Risk Mitigation
- Enhanced monitoring in high-risk zones
- Improved early warning systems
- Community preparedness programs
- Infrastructure resilience planning

---
*Analysis completed as part of 30 Days of Datasets*
*Model achieves 80-90% accuracy in tsunami risk prediction*
