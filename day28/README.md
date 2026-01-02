# Day 28: Meal & Food Data Analysis - Nutrition & Recommendations

## Project Overview

This project analyzes meal and food data to understand nutritional patterns, dietary preferences, and build a foundation for food recommendation systems. By integrating meal composition data with nutritional metadata, we provide insights for healthy eating, meal planning, and dietary optimization.

## Datasets

**Files:** 
1. `Final_data.csv` - Comprehensive meal/food records with consumption patterns
2. `meal_metadata.csv` - Nutritional information and meal characteristics

**Key Features:**
- **Meal Information:** Meal type (breakfast, lunch, dinner), cuisine category
- **Nutritional Data:** Calories, protein, carbs, fats, fiber, vitamins
- **Ingredients:** Main ingredients, allergen information
- **Metadata:** Preparation time, serving size, difficulty level
- **User Preferences:** Ratings, consumption frequency, dietary restrictions
- **Health Indicators:** Nutritional balance scores, health tags

## Objectives

1. **Nutritional Analysis:**
   - Caloric distribution across meal types
   - Macronutrient balance assessment (protein, carbs, fats)
   - Micronutrient content analysis
   - Meal type nutritional profiles (breakfast vs lunch vs dinner)

2. **Dietary Pattern Discovery:**
   - Cuisine type popularity and nutritional characteristics
   - Healthy vs indulgent meal clustering
   - Meal combination patterns
   - Temporal eating patterns (if timestamp data available)

3. **Multi-Dataset Integration:**
   - Join meal data with nutritional metadata
   - Enrich records with calculated metrics
   - Cross-reference ingredient lists with nutrition facts
   - Build comprehensive meal profiles

4. **Recommendation System Foundations:**
   - Meal similarity clustering
   - Nutritional goal-based filtering
   - Ingredient substitution analysis
   - Balanced meal plan generation

## Analysis Techniques

- Multi-source data integration and joining
- Nutritional scoring and classification
- Clustering for meal grouping (K-Means, DBSCAN)
- Recommendation algorithms (content-based filtering)
- Feature engineering (macro ratios, calorie density)
- Statistical analysis of dietary patterns
- Visualization of nutritional balance

## Expected Outcomes

- Comprehensive nutritional profile dashboard
- Meal type characteristic identification
- Cuisine-based nutritional insights
- Ingredient-nutrition correlation analysis
- Meal clustering by nutritional similarity
- Recommendation prototype for healthy alternatives
- Balanced meal plan templates
- Dietary guideline compliance scoring

## Visualizations

- Calorie distribution histograms by meal type
- Macronutrient ratio pie charts
- Nutritional balance radar charts
- Cuisine comparison box plots
- Ingredient frequency word clouds
- Meal cluster scatter plots (using PCA/t-SNE)
- Correlation heatmap of nutritional features
- Time-based consumption patterns

## Key Insights Expected

- Breakfast meals typically lower calorie than lunch/dinner
- Certain cuisines naturally more balanced nutritionally
- Protein content varies significantly by meal type
- Calorie density differs across preparation methods
- Ingredient combinations impact overall nutrition
- Healthy meals cluster together based on macro ratios
- Preparation time doesn't correlate strongly with health scores

## Nutritional Metrics

- **Calorie Density:** Calories per serving size
- **Macro Ratios:** Protein/Carb/Fat percentages
- **Nutrition Score:** Composite health score
- **Balance Index:** Distribution across food groups
- **Fiber Content:** Dietary fiber adequacy

## Project Structure

- `data/` - Two datasets (meal data and metadata) with integration scripts
- `models/` - Clustering models, recommendation algorithms, scoring systems
- `notebooks/` - Data integration, nutritional analysis, and recommendation notebooks
- `viz/` - Nutritional dashboards, meal comparisons, dietary pattern visualizations

## Getting Started

Start with the data integration notebook to understand how the two datasets connect, then explore the nutritional analysis notebook for comprehensive dietary insights, and finally review the recommendation prototype for meal planning ideas.
