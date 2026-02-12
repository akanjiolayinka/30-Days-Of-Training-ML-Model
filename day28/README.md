# Day 28: Fitness & Nutrition Tracking Analysis

**Project Type:** Hybrid Health Analytics (Fitness + Meal Tracking Integration)

## Overview
This project analyzes a comprehensive fitness and nutrition tracking dataset that combines workout metrics, exercise performance data, and meal logging. By integrating physical activity measurements with dietary intake records, we explore the relationships between exercise intensity, nutritional balance, and calorie management for health optimization.

## Dataset

**Files:**
1. `Final_data.csv` - 20,000 records combining fitness tracking with meal logs
2. `meal_metadata.csv` - 169 records with identical schema (subset for validation)

**Key Features:**

**Fitness Metrics:**
- Age, Gender, Weight, Height, BMI
- Heart Rate measurements (Resting, Average, Max BPM)
- Workout details (Type, Duration, Frequency, Experience Level)
- Calories Burned, Fat Percentage, Water Intake

**Exercise Data:**
- Exercise Name, Sets, Reps, Target Muscle Group
- Equipment Needed, Difficulty Level, Body Part
- Expected Calorie Burn per 30min

**Nutrition Tracking:**
- Meal name, type (Breakfast/Lunch/Dinner/Snack), diet type (Keto/Paleo/Vegan/Vegetarian/Balanced/Low-Carb)
- Macronutrients: Carbs, Proteins, Fats, Calories
- Micronutrients: Sugar, Sodium, Cholesterol
- Meal details: Serving size, cooking method, prep/cook time, rating

**Calculated Metrics:**
- Calorie balance (intake vs burned)
- Macro percentages, protein per kg body weight
- Heart rate reserve percentage
- Lean mass estimation

## Objectives

1. **Fitness Pattern Analysis:**
   - Workout type distribution and intensity
   - Heart rate profiles during exercise
   - Relationship between workout frequency and fitness metrics
   - Calorie burn patterns across workout types

2. **Nutrition Analysis:**
   - Diet type preferences and macro profiles
   - Meal type nutritional characteristics
   - Micronutrient intake assessment
   - Cooking method impact on nutrition

3. **Fitness-Nutrition Integration:**
   - Calorie balance analysis (deficit/surplus patterns)
   - Protein intake vs body weight relationship
   - Workout intensity correlation with dietary choices
   - Recovery nutrition patterns

4. **Predictive Modeling:**
   - Workout type classification based on fitness metrics
   - Diet type prediction using nutrition + fitness features
   - Calorie balance category prediction

## Analysis Performed

- Exploratory Data Analysis (EDA) with 9 visualizations
- Statistical analysis of fitness and nutrition distributions
- Correlation analysis between fitness metrics and nutrition
- Classification modeling (Random Forest)
- Feature importance analysis for workout and diet predictions

## Models

**Model 1: Workout Type Classification**
- Algorithm: Random Forest Classifier (n_estimators=100, max_depth=10)
- Features: Age, BMI, Heart Rate metrics, Fat %, Water Intake, Workout Frequency
- Target: Workout Type (Cardio / HIIT / Strength / Yoga)
- Accuracy: ~26% (baseline performance - features show limited direct prediction power)

**Model 2: Diet Type Classification**
- Algorithm: Random Forest Classifier (n_estimators=100, max_depth=10)
- Features: Age, BMI, Calories Burned, Macros, Calorie Balance
- Target: Diet Type (Balanced / Keto / Low-Carb / Paleo / Vegan / Vegetarian)
- Accuracy: ~16% (challenging multi-class problem with overlapping patterns)

**Note:** Low model accuracy indicates that workout type and diet type are not directly predictable from basic fitness/nutrition metrics alone - they reflect personal preference more than physiological constraints.

## Key Findings

- **Workout Distribution**: Balanced across types (Strength: 25%, Yoga: 25%, HIIT: 25%, Cardio: 25%)
- **Diet Preferences**: Paleo and Low-Carb most common (~17% each), followed by Vegan/Vegetarian/Keto/Balanced
- **Meal Patterns**: Relatively uniform across Breakfast/Lunch/Dinner/Snack
- **Calorie Balance**: Wide range from large deficits to large surpluses
- **Heart Rate**: Max BPM typically 170-190, Average 130-160, Resting 50-75
- **Macros by Meal**: Lunch and Dinner contain higher protein/carbs than Breakfast on average
- **Correlations**: Weak correlations between workout type and diet type - indicates independent choices

## Files

```
day28/
├── data/
│   ├── Final_data.csv           # Main dataset (20,000 records, 54 features)
│   └── meal_metadata.csv        # Validation subset (169 records)
├── notebooks/
│   └── run_analysis.py          # Complete analysis script
├── models/
│   ├── workout_type_rf_model.joblib
│   ├── diet_type_rf_model.joblib
│   ├── workout_scaler.joblib
│   ├── diet_scaler.joblib
│   ├── workout_label_encoder.joblib
│   └── diet_label_encoder.joblib
├── viz/
│   ├── workout_diet_distribution.png
│   ├── calories_by_workout.png
│   ├── fitness_metrics_scatter.png
│   ├── calorie_balance_distribution.png
│   ├── meal_type_macros.png
│   ├── heart_rate_distributions.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

## Usage

Run the analysis script to generate all visualizations and train models:

```bash
cd day28/notebooks
python run_analysis.py
```

This will:
1. Load and clean the fitness + nutrition dataset
2. Generate 9 visualization files (saved to viz/)
3. Train workout type and diet type classification models
4. Evaluate model performance with confusion matrices
5. Analyze feature importance
6. Save all 6 model artifacts to models/

## Potential Extensions

- Time-series analysis if temporal data added
- Clustering to identify fitness-nutrition personas
- Regression to predict calorie balance outcomes
- Recommendation system for balanced nutrition based on workout intensity
- Anomaly detection for extreme fitness/nutrition patterns
- Causal analysis: Does workout type influence diet choice or vice versa?

## Author
**Olayinka Akanji**
