# Day 21: Data Science Job Salaries Analysis & Prediction

## Project Overview

This project analyzes the data science job market to understand salary dynamics across different roles, experience levels, company sizes, and geographic locations. By examining real-world salary data from 2020-2023, we build predictive models to estimate compensation and provide insights for both job seekers and employers in the data science field.

## Dataset

**File:** `ds_salaries.csv`

**Description:** Data science salaries dataset with 600+ records from various countries and job roles

**Key Features:**

**Temporal:**
- Work year (2020-2023)

**Experience & Employment:**
- Experience level (EN: Entry, MI: Mid, SE: Senior, EX: Executive)
- Employment type (FT: Full-time, PT: Part-time, CT: Contract, FL: Freelance)

**Job Information:**
- Job title (Data Scientist, ML Engineer, Data Analyst, etc.)
- Remote ratio (0%, 50%, 100%)

**Compensation:**
- Salary (in original currency)
- Salary currency (EUR, USD, GBP, etc.)
- Salary in USD (standardized)

**Location:**
- Employee residence (country code)
- Company location (country code)

**Company:**
- Company size (S: Small, M: Medium, L: Large)

## Objectives

1. **Salary Distribution Analysis:**
   - Average salaries by job title
   - Experience level vs compensation correlation
   - Geographic salary variations (USD standardized)
   - Company size impact on salaries
   - Remote work vs on-site salary differences

2. **Job Market Trends:**
   - Salary growth year-over-year (2020-2023)
   - Most in-demand roles and their compensation
   - Entry vs senior level salary gaps
   - Regional market competitiveness
   - Remote work adoption trends

3. **Predictive Modeling:**
   - Salary prediction using regression models:
     - Linear Regression (baseline)
     - Random Forest Regressor
     - Gradient Boosting (XGBoost, LightGBM)
     - Ridge/Lasso Regression
   - Feature importance for salary determination
   - Model evaluation (R², MAE, RMSE, MAPE)

4. **Career Insights:**
   - Highest paying roles by experience level
   - Career progression salary trajectory
   - Remote work compensation analysis
   - Geographic arbitrage opportunities
   - Company size vs compensation trade-offs

## Analysis Techniques

- Regression modeling with cross-validation
- Feature engineering (experience encoding, location grouping)
- Currency standardization (all to USD)
- Categorical encoding (one-hot, target encoding)
- Time series analysis for salary trends
- Geographic analysis and mapping
- Statistical hypothesis testing

## Expected Outcomes

- Predictive model with R² > 0.75
- Experience level as strongest salary predictor
- Job title second most important factor
- Company size shows 20-40% salary difference (L vs S)
- Remote work analysis (100% remote vs on-site)
- Geographic salary map (US, EU, Asia comparisons)
- Yearly salary growth trends (2020-2023)
- Top 10 highest paying data science roles
- Entry to Executive salary progression curve

## Visualizations

- Salary distribution histograms by experience level
- Box plots comparing salaries across job titles
- Geographic heatmap of average salaries by country
- Time series: Yearly salary trends (2020-2023)
- Remote ratio vs salary scatter plots
- Company size comparison bar charts
- Feature importance rankings
- Actual vs Predicted salary scatter plots
- Correlation heatmap of numerical features
- Top 20 job titles by average salary

## Key Insights Expected

- **Experience Level:** Dominates salary determination
  - Entry (EN): $40K-80K
  - Mid (MI): $70K-120K
  - Senior (SE): $100K-180K
  - Executive (EX): $150K-300K+

- **Top Paying Roles:**
  - Machine Learning Engineer
  - Data Science Manager
  - Principal Data Scientist
  - ML Architect
  - Research Scientist

- **Geographic Differences:**
  - US salaries typically 20-40% higher than Europe
  - Switzerland and Germany lead European markets
  - Asia shows wide variation by country

- **Remote Work:**
  - 100% remote roles competitive with on-site
  - Pandemic (2020-2021) accelerated remote adoption
  - Geographic arbitrage opportunities

- **Company Size:**
  - Large companies (L) pay 25-35% more than small (S)
  - Medium companies (M) offer middle ground

- **Salary Growth:**
  - Year-over-year increase of 5-15% (2020-2023)
  - Pandemic disruption visible in 2020-2021
  - Market stabilization in 2022-2023

## Feature Engineering

- **Experience Encoding:** Ordinal encoding (EN=0, MI=1, SE=2, EX=3)
- **Location Categories:** Group by region (Americas, Europe, Asia, etc.)
- **Remote Categories:** Bins (0%, 50%, 100%)
- **Job Title Grouping:** Categorize similar roles
- **Year Features:** Extract trends, year-over-year changes

## Machine Learning Pipeline

```python
# Salary Prediction Pipeline
1. Data Loading → 2. Currency Standardization (USD)
→ 3. Feature Engineering → 4. Encoding & Scaling
→ 5. Train/Test Split → 6. Model Training
→ 7. Evaluation → 8. Salary Estimation Tool
```

## Model Comparison

| Model | Expected R² | MAE | Best For |
|-------|------------|-----|----------|
| Linear Regression | 0.70 | $25K | Baseline |
| Random Forest | 0.80 | $18K | Feature Importance |
| XGBoost | 0.83 | $15K | Best Performance |
| LightGBM | 0.82 | $16K | Fast Training |

## Practical Applications

- **Job Seekers:** Salary expectations by role and experience
- **Employers:** Competitive compensation benchmarking
- **Career Planning:** Understand salary progression paths
- **Negotiation:** Data-driven salary discussions
- **Remote Work:** Geographic salary arbitrage insights

## Project Structure

- `data/` - Data science salaries dataset (2020-2023)
- `models/` - Trained regression models and salary prediction pipelines
- `notebooks/` - EDA, market analysis, and predictive modeling notebooks
- `viz/` - Salary trends, geographic maps, role comparisons, career trajectories

## Getting Started

Begin with the EDA notebook to understand the data science job market landscape, then explore the modeling notebook to build a salary prediction tool that helps estimate compensation based on experience, role, and location.
