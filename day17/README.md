# Day 17: Employee Attrition & HR Analytics

## Project Overview

This project analyzes employee attrition patterns using IBM's HR dataset to predict which employees are likely to leave the company. By examining demographic factors, job satisfaction metrics, compensation, work-life balance, and career progression indicators, we build predictive models to help HR departments identify at-risk employees and implement retention strategies.

## Dataset

**File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`

**Description:** Comprehensive IBM HR employee records with 1,470 samples and 35 features

**Key Features:**

**Demographic:**
- Age, Gender, Marital Status
- Education, Education Field
- Distance from home

**Job Information:**
- Job Role, Job Level, Department
- Years at company, Years in current role
- Years since last promotion
- Years with current manager

**Compensation:**
- Monthly income, Monthly rate, Daily rate, Hourly rate
- Percent salary hike
- Stock option level

**Satisfaction & Work-Life:**
- Job satisfaction, Environment satisfaction
- Relationship satisfaction, Work-life balance
- Job involvement, Performance rating

**Work Patterns:**
- Business travel frequency, Overtime status
- Total working years, Training times last year
- Number of companies worked

**Target Variable:** Attrition (Yes/No - Binary Classification)

## Objectives

1. **Attrition Pattern Analysis:**
   - Overall attrition rate calculation
   - Attrition by department, job role, and age group
   - Demographic factors correlation with turnover
   - Compensation vs attrition relationship

2. **Risk Factor Identification:**
   - Which factors most predict employee departure?
   - Job satisfaction impact on retention
   - Work-life balance vs attrition
   - Career progression (promotions, role changes) effects
   - Overtime and travel burden analysis

3. **Predictive Modeling:**
   - Binary classification for attrition prediction:
     - Logistic Regression (baseline)
     - Random Forest Classifier
     - Gradient Boosting (XGBoost, LightGBM)
     - Support Vector Machine (SVM)
   - Handle class imbalance (likely more non-attrition cases)
   - Feature importance ranking

4. **HR Strategy Recommendations:**
   - High-risk employee profiling
   - Retention intervention points
   - Compensation adjustment insights
   - Work-life balance improvement areas

## Analysis Techniques

- Binary classification with imbalanced data handling
- SMOTE (Synthetic Minority Over-sampling) if needed
- Feature engineering (tenure ratios, satisfaction scores)
- Statistical hypothesis testing (chi-square, t-tests)
- Survival analysis for employee tenure
- Cross-validation with stratified sampling
- ROC-AUC optimization for imbalanced classes

## Expected Outcomes

- Attrition prediction model with >80% accuracy and high recall
- Identification of top 5-10 attrition risk factors
- Job satisfaction and work-life balance as key predictors
- Overtime status strongly correlated with attrition
- Years since last promotion critical threshold (>3 years risk)
- Distance from home impact analysis
- Monthly income vs attrition inverse relationship
- Department-specific attrition patterns (Sales typically higher)
- Age groups most prone to leaving (younger employees)

## Visualizations

- Attrition rate by department, role, and demographics
- Salary distribution for attrition vs retention
- Correlation heatmap of all features
- Feature importance bar charts
- Job satisfaction vs attrition box plots
- Years at company distribution by attrition status
- Overtime and travel impact charts
- ROC curves and precision-recall curves
- Confusion matrix with focus on recall (catching leavers)

## Key Insights Expected

- **Overtime workers** have 2-3x higher attrition rate
- **Job satisfaction** score <3 indicates high risk
- **Work-life balance** critical for retention
- **Distance from home** >15km increases attrition
- **Years since last promotion** >4 years = flight risk
- **Monthly income** inversely correlated with leaving
- **Frequent business travel** contributes to attrition
- **Single employees** more likely to leave than married
- **Sales department** typically highest attrition
- **Stock options** improve retention significantly

## Machine Learning Pipeline

```python
# HR Analytics Pipeline
1. Data Loading & Cleaning → 2. EDA & Statistical Tests
→ 3. Feature Engineering → 4. Class Imbalance Handling
→ 5. Model Training → 6. Evaluation (Focus on Recall)
→ 7. Risk Scoring System
```

**Special Considerations:**
- **Class Imbalance:** Handle with SMOTE, class weights, or threshold tuning
- **Recall Optimization:** Minimize false negatives (missing at-risk employees)
- **Interpretability:** Use explainable models for HR actionability

## Retention Risk Scoring

**High Risk Indicators (Weighted):**
- Overtime: Yes (30%)
- Job Satisfaction: 1-2 (25%)
- Years Since Promotion: >3 (20%)
- Work-Life Balance: 1-2 (15%)
- Monthly Income: Below median (10%)

## Model Evaluation Metrics

- **Accuracy:** Overall correctness
- **Recall (Sensitivity):** Most critical - catch employees who will leave
- **Precision:** Avoid false alarms
- **F1-Score:** Balance between precision and recall
- **ROC-AUC:** Model discrimination ability
- **Confusion Matrix:** Detailed error analysis

## Project Structure

- `data/` - IBM HR employee attrition dataset
- `models/` - Trained classifiers with class imbalance handling
- `notebooks/` - EDA, statistical analysis, and predictive modeling notebooks
- `viz/` - Attrition dashboards, factor analysis, risk profiling charts

## Getting Started

Begin with the EDA notebook to understand attrition patterns and key risk factors, then explore the modeling notebook to build a predictive system that identifies employees at risk of leaving.
