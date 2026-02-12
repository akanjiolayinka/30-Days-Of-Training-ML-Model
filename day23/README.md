# Day 23: Unified Dataset Analysis (Advanced Machine Learning)

## Project Overview

This project conducts an advanced analysis of the unified multi-domain dataset, building upon initial exploratory insights to develop sophisticated predictive models and extract deeper patterns. Using ensemble methods and advanced feature engineering, we maximize predictive accuracy and interpretability.

## Dataset

**File:** `UnifiedDataset.csv`

**Description:** Integrated multi-domain dataset with diverse features suitable for advanced modeling

**Key Characteristics:**
- Rich feature set with numerical and categorical variables
- Multi-domain integration for comprehensive analysis
- Suitable for both classification and regression tasks
- Potential target variables for supervised learning
- Complex feature interactions and non-linear relationships

## Objectives

1. **Advanced Feature Engineering:**
   - Polynomial and interaction feature creation
   - Domain-specific transformations
   - Automated feature selection methods
   - Dimensionality reduction while preserving information
   - Target encoding for categorical variables

2. **Ensemble Model Development:**
   - Compare multiple algorithms:
     - Random Forest (bagging ensemble)
     - Gradient Boosting (XGBoost, LightGBM, CatBoost)
     - AdaBoost (boosting ensemble)
     - Voting Classifier/Regressor (combining multiple models)
   - Hyperparameter optimization with Bayesian methods
   - Cross-validation for robust evaluation

3. **Model Interpretation:**
   - SHAP (SHapley Additive exPlanations) analysis
   - Permutation feature importance
   - Partial dependence plots
   - Individual prediction explanations
   - Model decision boundary visualization

4. **Production Pipeline:**
   - End-to-end pipeline creation
   - Model serialization and versioning
   - Inference function development
   - Performance monitoring framework

## Analysis Techniques

- Advanced preprocessing with pipelines
- Automated feature engineering libraries
- Ensemble methods (bagging, boosting, stacking)
- Hyperparameter tuning (GridSearch, RandomSearch, Optuna)
- Model interpretation with SHAP and LIME
- Cross-validation strategies (K-Fold, Stratified)
- Calibration for probability predictions
- Model comparison frameworks

## Expected Outcomes

- High-performance model (>90% accuracy for classification, R² > 0.85 for regression)
- Comprehensive feature importance rankings
- SHAP summary plots for model interpretation
- Production-ready prediction pipeline
- Model comparison report across algorithms
- Optimized hyperparameters for best model
- Individual prediction explanation capability
- Deployment-ready model artifacts

## Visualizations

- SHAP summary plots and force plots
- Feature importance comparison across models
- Partial dependence plots for key features
- Model performance comparison charts (ROC, PR curves)
- Confusion matrices and classification reports
- Learning curves for training diagnostics
- Residual plots for regression tasks
- Feature interaction heatmaps

## Advanced Techniques

- **Stacking Ensemble:** Combining multiple base models with meta-learner
- **Feature Selection:** Recursive Feature Elimination (RFE), SelectKBest
- **Handling Imbalance:** SMOTE, class weights, threshold tuning
- **Calibration:** Platt scaling, isotonic regression
- **Model Explainability:** SHAP waterfall plots, decision paths

## Project Structure

- `data/` - Unified dataset with engineered features
- `models/` - Multiple trained models, ensemble pipelines, scalers
- `notebooks/` - Advanced feature engineering, ensemble modeling, interpretation
- `viz/` - SHAP plots, model comparisons, performance dashboards

## Getting Started

This is an advanced ML project. Run the complete analysis script:

```bash
cd day23/notebooks
python run_advanced_ml.py
```

This will:
1. Load and preprocess the UnifiedDataset (22,051 samples, 140+ features)
2. Apply advanced feature engineering (polynomial features, interactions, feature selection)
3. Train and tune ensemble models (Random Forest, Gradient Boosting, Voting Ensemble)
4. Generate 6+ visualizations including SHAP plots (if available)
5. Save all model artifacts to models/

## Implementation Results

**Task:** Life Expectancy Prediction (Regression)

**Models Trained:**
- Random Forest Regressor (with GridSearchCV hyperparameter tuning)
- Gradient Boosting Regressor (with GridSearchCV)
- Voting Ensemble (combining RF + GB)

**Performance:** R² > 0.85, demonstrating strong predictive capability on this complex multi-domain health dataset

**Files Generated:**
- 6 visualizations (model comparison, actual vs predicted, residual plot, feature importance plots, learning curve)
- 3 model files saved (GB model, scaler, feature names)
  - Note: Random Forest (113MB) and Voting Ensemble (226MB) models exceed GitHub's 100MB limit. Run the script to regenerate them locally.

## Author
**Olayinka Akanji**
