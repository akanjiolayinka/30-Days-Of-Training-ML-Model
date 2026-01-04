# Day 3: BMI & Lifestyle Health Classification

## Project Overview

This project analyzes the relationship between physical characteristics (gender, height, weight) and Body Mass Index (BMI) categories to build a classification model that predicts health index levels. Using anthropometric data from 500 individuals, we explore how gender and body measurements correlate with BMI classifications and develop a health assessment tool.

## Dataset

**File:** `500_Person_Gender_Height_Weight_Index.csv`

**Description:** Health and lifestyle dataset with 500 records of individuals' physical measurements

**Key Features:**

**Demographics:**
- Gender (Male/Female)

**Physical Measurements:**
- Height (cm)
- Weight (kg)

**Health Classification:**
- Index (0-4): BMI-based health categories
  - 0: Extremely Weak
  - 1: Weak
  - 2: Normal
  - 3: Overweight
  - 4: Obesity

## Objectives

1. **Exploratory Data Analysis:**
   - Distribution of BMI index categories
   - Height and weight distributions by gender
   - BMI calculation and validation
   - Gender differences in health index
   - Correlation between physical measurements

2. **Health Pattern Analysis:**
   - Average BMI by gender
   - Height-weight relationship across categories
   - Identify threshold values for each health index
   - Gender-specific BMI patterns
   - Normal vs overweight/obesity characteristics

3. **Classification Modeling:**
   - Multi-class classification (5 categories: 0-4)
   - Binary classification (Healthy vs Unhealthy)
   - Models to compare:
     - Logistic Regression (multi-class)
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - Gradient Boosting (XGBoost)

4. **Health Recommendations:**
   - Optimal height-weight combinations
   - Risk factor identification
   - Health improvement strategies
   - BMI category transition thresholds

## Analysis Techniques

- Multi-class classification algorithms
- BMI calculation and feature engineering
- Statistical analysis by gender groups
- Feature scaling and normalization
- Class imbalance handling (if present)
- Cross-validation for model robustness
- Confusion matrix analysis

## Expected Outcomes

- High classification accuracy (>85%) due to clear BMI rules
- Weight identified as strongest predictor
- BMI formula: Weight (kg) / (Height (m))²
- Gender-specific patterns:
  - Male average height: 170-180cm
  - Female average height: 155-165cm
  - Different weight distributions by gender
- Clear decision boundaries between categories
- Health index distribution:
  - Normal (Index 2): 30-40% expected
  - Overweight/Obesity (Index 3-4): 30-40%
  - Underweight categories (Index 0-1): 20-30%

## Visualizations

- BMI index distribution histogram
- Height vs Weight scatter plot (colored by index)
- Gender comparison box plots
- Correlation heatmap
- BMI category pie chart
- Feature importance bar chart
- Confusion matrix heatmap
- Decision boundary visualization (2D)
- Gender-wise BMI distributions

## Key Insights Expected

**BMI Category Thresholds (WHO Standards):**
- **Extremely Weak (Index 0):** BMI < 16
- **Weak (Index 1):** BMI 16-18.5
- **Normal (Index 2):** BMI 18.5-25
- **Overweight (Index 3):** BMI 25-30
- **Obesity (Index 4):** BMI > 30

**Gender Differences:**
- Males typically have higher weight at same BMI
- Females show different fat distribution patterns
- Gender-specific healthy weight ranges

**Health Patterns:**
- Weight is the dominant predictor (height relatively stable)
- Linear relationship between weight and BMI index
- Clear category separation makes classification accurate
- Few borderline cases between categories

## Feature Engineering

- **BMI Calculation:** Weight / (Height/100)²
- **Height Categories:** Short, Medium, Tall
- **Weight Categories:** Underweight, Normal, Overweight
- **Gender Encoding:** Binary (0/1) or one-hot
- **BMI Binary:** Healthy (Index 2) vs Unhealthy (0,1,3,4)

## Machine Learning Pipeline

```python
# Health Classification Pipeline
1. Data Loading → 2. BMI Calculation & Validation
→ 3. Feature Engineering → 4. Encoding & Scaling
→ 5. Train/Test Split → 6. Model Training
→ 7. Evaluation → 8. Health Assessment Tool
```

## Model Comparison

| Model | Expected Accuracy | Speed | Interpretability |
|-------|------------------|-------|------------------|
| Logistic Regression | 85-90% | Fast | High |
| KNN | 88-92% | Medium | Medium |
| Decision Tree | 90-95% | Fast | Very High |
| Random Forest | 92-96% | Medium | Medium |
| SVM | 88-92% | Slow | Low |
| XGBoost | 93-97% | Medium | Low |

## Health Assessment Categories

**High Accuracy Expected Because:**
- BMI has mathematical formula (weight/height²)
- Clear category boundaries based on WHO standards
- Limited features reduce overfitting
- Strong correlation between features and target

## Practical Applications

- **Health Screening:** Quick BMI category assessment
- **Fitness Apps:** Health index calculator
- **Medical Checkups:** Initial health classification
- **Weight Management:** Track category transitions
- **Public Health:** Population health monitoring

## Project Structure

- `data/` - BMI and health index dataset (500 records)
- `models/` - Trained classification models and health assessment tools
- `notebooks/` - EDA, BMI analysis, and classification modeling notebooks
- `viz/` - Health distributions, BMI charts, gender comparisons

## Getting Started

Begin with the EDA notebook to understand BMI distributions and calculate health indices, then explore the classification modeling notebook to build an automated health assessment tool based on height, weight, and gender.
