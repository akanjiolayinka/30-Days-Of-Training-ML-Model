# Day 27: Wine Quality Classification & Analysis

## Project Overview

This project analyzes red wine quality based on physicochemical properties to predict wine ratings and understand which chemical characteristics contribute to superior wine quality. Using laboratory test results including acidity, sugar content, pH levels, and alcohol percentage, we build classification models to help winemakers optimize production and sommeliers assess wine quality.

## Dataset

**File:** `winequality-red.csv`

**Description:** Red wine quality dataset with 1,600 samples and 11 physicochemical features

**Key Features:**

**Acidity Measures:**
- Fixed acidity (g/dm³) - tartaric acid concentration
- Volatile acidity (g/dm³) - acetic acid (vinegar taste)
- Citric acid (g/dm³) - freshness and flavor
- pH - acidity/basicity scale (0-14)

**Sweetness & Preservation:**
- Residual sugar (g/dm³) - remaining sugar after fermentation
- Free sulfur dioxide (mg/dm³) - prevents microbial growth
- Total sulfur dioxide (mg/dm³) - free + bound SO₂

**Physical Properties:**
- Density (g/cm³) - wine mass per volume
- Alcohol (% by volume) - ethanol content

**Mineral Content:**
- Chlorides (g/dm³) - salt content
- Sulphates (g/dm³) - antimicrobial and antioxidant

**Target Variable:** Quality (score: 0-10, typically 3-8 in dataset)

## Objectives

1. **Wine Chemistry Analysis:**
   - Distribution of quality scores
   - Physicochemical properties correlation with quality
   - Identify optimal ranges for high-quality wines
   - Alcohol content vs quality relationship
   - Acidity balance importance

2. **Quality Drivers Identification:**
   - Which chemical properties most influence quality?
   - Alcohol percentage impact on ratings
   - Volatile acidity (vinegar taste) negative correlation
   - Sulphate and citric acid positive effects
   - pH and acidity balance optimal zones

3. **Classification Modeling:**
   - Multi-class classification (quality scores 3-8)
   - Binary classification (high vs low quality)
   - Regression for continuous quality prediction
   - Models to compare:
     - Random Forest Classifier/Regressor
     - Gradient Boosting (XGBoost, LightGBM)
     - Support Vector Machine (SVM)
     - Neural Networks (MLP)

4. **Winemaking Insights:**
   - Optimal chemical composition for quality wines
   - Warning indicators for poor quality
   - Feature importance for quality control
   - Recommendations for production optimization

## Analysis Techniques

- Multi-class and binary classification
- Regression for quality score prediction
- Feature scaling and normalization
- Correlation analysis and heatmaps
- Feature engineering (acidity ratios, balance indices)
- Cross-validation for robust evaluation
- Hyperparameter tuning
- Model interpretation and SHAP analysis

## Expected Outcomes

- Classification accuracy 65-75% (quality is subjective)
- Regression model R² 0.40-0.60 (moderate due to subjectivity)
- **Alcohol** identified as top quality predictor (positive)
- **Volatile acidity** second most important (negative)
- **Sulphates** and **citric acid** positive quality indicators
- **Residual sugar** minimal impact on red wine quality
- Quality scores concentrated in 5-6 range (normal distribution)
- High-quality wines (7-8) show:
  - Higher alcohol (>11%)
  - Lower volatile acidity (<0.6)
  - Higher sulphates (>0.7)
  - Moderate fixed acidity (7-9)

## Visualizations

- Quality score distribution histogram
- Correlation heatmap of all chemical properties
- Box plots: Chemical features by quality score
- Scatter plots (Alcohol vs Quality, Volatile Acidity vs Quality)
- Pair plots for key feature relationships
- Feature importance bar charts
- Confusion matrix for classification
- Residual plots for regression
- 3D scatter plots (alcohol-acidity-quality)
- Quality category distributions

## Key Insights Expected

### Positive Quality Indicators:
- **High Alcohol (11-14%):** Strongly correlated with quality
- **Moderate-High Sulphates (0.6-1.0):** Preserves wine, adds complexity
- **Moderate Citric Acid (0.2-0.5):** Adds freshness
- **Low Volatile Acidity (<0.5):** Prevents vinegar taste
- **Balanced pH (3.0-3.5):** Ideal acidity range

### Negative Quality Indicators:
- **High Volatile Acidity (>0.8):** Vinegar taste, quality drops
- **Low Alcohol (<9%):** Correlated with lower ratings
- **Very Low Citric Acid (<0.1):** Lack of freshness
- **Extreme pH values:** Too acidic or too basic

### Neutral Factors:
- **Residual Sugar:** Minimal impact on red wine quality
- **Chlorides:** Slight negative if too high
- **Density:** Correlated with alcohol (inverse)

## Quality Categories (Binary Classification)

**Low Quality (3-5):** 50-60% of samples
**High Quality (6-8):** 40-50% of samples

Binary classification typically achieves 70-80% accuracy.

## Feature Engineering

- **Acidity Balance:** Fixed acidity / Volatile acidity ratio
- **Total Acidity:** Sum of fixed, volatile, and citric acid
- **SO₂ Ratio:** Free SO₂ / Total SO₂
- **Alcohol Category:** Bins (low, medium, high)
- **Quality Binary:** High (≥6) vs Low (<6)
- **Chemical Balance Score:** Composite of optimal ranges

## Machine Learning Pipeline

```python
# Wine Quality Prediction Pipeline
1. Data Loading & Cleaning → 2. EDA & Correlation Analysis
→ 3. Feature Engineering → 4. Scaling & Encoding
→ 5. Model Training (Classification & Regression)
→ 6. Evaluation → 7. Quality Prediction Tool
```

## Model Performance Expectations

| Approach | Model | Expected Accuracy/R² |
|----------|-------|---------------------|
| Multi-class (3-8) | Random Forest | 60-70% |
| Binary (High/Low) | XGBoost | 75-82% |
| Regression | Gradient Boosting | R² 0.45-0.55 |

**Note:** Quality is partially subjective, so perfect prediction is challenging.

## Practical Applications

- **Winemakers:** Optimize chemical composition for quality
- **Quality Control:** Identify batches likely to score high/low
- **Sommeliers:** Understand chemical basis of wine quality
- **Wine Education:** Learn quality drivers scientifically
- **Production:** Adjust fermentation and aging processes

## Domain Knowledge Integration

- **Fermentation:** Alcohol produced from sugar
- **Oxidation:** Volatile acidity increases over time if exposed to air
- **Preservation:** SO₂ critical for wine longevity
- **Balance:** Great wines have harmony among components
- **Terroir:** Chemical properties reflect grape origin (not in dataset)

## Project Structure

- `data/` - Red wine quality dataset with physicochemical properties
- `models/` - Classification and regression models for quality prediction
- `notebooks/` - Chemistry analysis, feature engineering, and modeling notebooks
- `viz/` - Chemical correlations, quality distributions, feature importance charts

## Getting Started

Start with the EDA notebook to understand wine chemistry and quality relationships, then explore the modeling notebook to build predictive models that classify wine quality based on laboratory measurements.
