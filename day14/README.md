# Day 14: Mobile Phone Price Classification

## Project Overview

This project builds a multi-class classification model to predict mobile phone price ranges based on hardware specifications and features. Using a comprehensive set of phone attributes including battery power, RAM, camera specifications, and connectivity features, we classify phones into different price categories to help consumers and retailers understand pricing dynamics.

## Datasets

**Files:** 
1. `train.csv` - Training data with 2,000 samples and price range labels
2. `test.csv` - Test data with 1,000 samples for predictions

**Key Features:**
- **Battery:** Battery power (mAh), talk time
- **Display:** Pixel height, pixel width, screen height, screen width, touch screen
- **Camera:** Front camera (MP), primary camera (MP)
- **Memory:** RAM (MB), internal memory (GB)
- **Processor:** Clock speed (GHz), number of cores
- **Connectivity:** 3G, 4G, Bluetooth, WiFi, dual SIM
- **Physical:** Mobile weight (g), mobile depth (cm)
- **Target:** Price range (0: Low, 1: Medium, 2: High, 3: Very High)

## Objectives

1. **Exploratory Data Analysis:**
   - Feature distribution across price ranges
   - Correlation between specifications and pricing
   - Identify key differentiators between price categories
   - Battery power vs RAM relationship analysis

2. **Feature Importance:**
   - Which specs most influence price range?
   - RAM vs processor speed impact comparison
   - Camera quality correlation with pricing
   - Connectivity features value assessment

3. **Multi-Class Classification:**
   - Build classification models:
     - Logistic Regression (multi-class)
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - Gradient Boosting (XGBoost, LightGBM)
     - K-Nearest Neighbors (KNN)
   - Cross-validation for robust evaluation
   - Confusion matrix analysis per price range

4. **Model Optimization:**
   - Hyperparameter tuning for best performance
   - Feature selection and engineering
   - Class imbalance handling (if present)
   - Ensemble methods for improved accuracy

## Analysis Techniques

- Multi-class classification algorithms
- Feature scaling and normalization
- Correlation analysis and feature selection
- Cross-validation (K-Fold, Stratified)
- Hyperparameter tuning (GridSearch, RandomSearch)
- Confusion matrix and classification metrics
- Feature importance ranking

## Expected Outcomes

- High-accuracy classification model (>85% accuracy target)
- RAM identified as top price predictor
- Battery power strongly correlated with price range
- Camera specs (primary camera) significant for high-end phones
- 4G and connectivity features important for premium segments
- Clear decision boundaries between price ranges
- Production-ready classification pipeline
- Price range prediction tool for new phone specs

## Visualizations

- Feature distribution box plots by price range
- Correlation heatmap of all specifications
- Confusion matrix for model evaluation
- Feature importance bar charts
- Scatter plots (RAM vs Battery, colored by price range)
- Pair plots showing key feature relationships
- ROC curves (one-vs-rest for each price range)
- Decision boundary visualization (using PCA)

## Key Insights Expected

- **RAM** is the strongest predictor of price range
- **Battery power** correlates with premium phones
- **Primary camera** quality separates mid from high-end
- **4G connectivity** standard in medium+ price ranges
- **Touch screen** presence in most price ranges
- **Pixel resolution** impacts high-end pricing
- **Weight** minimal correlation with price
- **Dual SIM** feature distributed across all ranges

## Machine Learning Pipeline

```python
# Classification Pipeline
1. Data Loading & EDA → 2. Feature Engineering → 3. Preprocessing
→ 4. Model Training → 5. Evaluation → 6. Prediction on Test Set
```

**Pipeline Components:**
- Missing value check (none expected)
- Feature scaling (StandardScaler)
- Feature selection (if needed)
- Multi-class classifier training
- Cross-validation
- Test set predictions

## Model Comparison

| Model | Expected Accuracy | Training Time | Interpretability |
|-------|------------------|---------------|------------------|
| Logistic Regression | 85-90% | Fast | High |
| Random Forest | 90-95% | Medium | Medium |
| XGBoost | 92-97% | Medium | Low |
| SVM | 88-93% | Slow | Low |
| KNN | 85-90% | Fast | Medium |

## Project Structure

- `data/` - Training and test datasets for mobile phone pricing
- `models/` - Trained classifiers and preprocessing pipelines
- `notebooks/` - EDA, feature analysis, and model training notebooks
- `viz/` - Specification charts, price comparisons, model performance plots

## Getting Started

Start with the EDA notebook to understand feature distributions across price ranges, then explore the classification modeling notebook to train and compare different algorithms for optimal price range prediction.
