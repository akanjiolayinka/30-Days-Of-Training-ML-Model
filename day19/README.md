# Day 19: Student Performance Factors Analysis

## Project Overview

This project investigates the various factors influencing student academic performance. By analyzing demographic characteristics, study habits, family background, and school-related variables, we identify key predictors of student success and provide actionable insights for educational improvement.

## Dataset

**File:** `StudentsPerformance.csv`

**Description:** Comprehensive student records with demographic, behavioral, and performance data

**Key Features:**
- **Demographics:** Gender, ethnicity, age
- **Family Background:** Parental education level, family income
- **Academic:** Math score, reading score, writing score
- **Study Habits:** Hours studied per week, attendance rate, homework completion
- **School Factors:** Class size, teaching quality indicators
- **Support Systems:** Tutoring, parental involvement, extracurricular activities

## Objectives

1. **Performance Analysis:**
   - Distribution of exam scores (math, reading, writing)
   - Average performance by demographic groups
   - Correlation between different subject scores
   - High performer vs low performer characteristics

2. **Factor Identification:**
   - Study hours impact on academic achievement
   - Attendance rate correlation with performance
   - Parental education influence on student outcomes
   - Gender differences in subject performance
   - Socioeconomic factors analysis

3. **Predictive Modeling:**
   - Multi-output regression to predict exam scores
   - Classification of performance categories (high/medium/low)
   - Feature importance for each subject area
   - Model comparison (Linear Regression, Random Forest, Gradient Boosting)

4. **Educational Insights:**
   - Intervention point identification
   - At-risk student profiling
   - Resource allocation recommendations
   - Best practices for improving outcomes

## Analysis Techniques

- Multi-output regression for simultaneous score prediction
- Statistical hypothesis testing (t-tests, ANOVA)
- Correlation and covariance analysis
- Feature engineering (average scores, performance categories)
- Classification and regression trees
- Cross-validation for model robustness
- Segment analysis by demographic groups

## Expected Outcomes

- Attendance and study hours identified as top predictors
- Positive correlation between parental education and scores
- Gender performance gap analysis across subjects
- Predictive model with reasonable accuracy (R² > 0.60)
- Risk factor identification for struggling students
- Data-driven recommendations for educators
- Interactive dashboard for performance monitoring
- Student success probability calculator

## Visualizations

- Score distribution histograms by subject
- Correlation heatmap of all variables
- Box plots comparing performance across groups
- Scatter plots (Study Hours vs Scores, Attendance vs Performance)
- Pair plots for multi-subject analysis
- Feature importance rankings
- Performance category pie charts
- Group comparison violin plots

## Key Insights Expected

- Attendance rate shows strong positive correlation with scores
- Hours studied per week significantly impact performance
- Parental education level influences student achievement
- Minimal gender differences in overall performance
- Homework completion rate is a critical success factor
- Early intervention can improve at-risk student outcomes

## Project Structure

- `data/` - Student performance dataset
- `models/` - Multi-output regression and classification models
- `notebooks/` - EDA, statistical analysis, and predictive modeling notebooks
- `viz/` - Performance charts, factor analysis plots, comparison dashboards

## Getting Started

Begin with the EDA notebook to explore performance distributions and factor relationships, then review the modeling notebook to understand which variables best predict student success.
