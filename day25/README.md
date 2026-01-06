# Day 25: Iris Flower Classification - Classic Machine Learning

**Author:** Olayinka Akanji

## Project Overview

This project tackles the classic Iris flower classification problem, one of the most well-known datasets in machine learning. Using four simple flower measurements, we build models to accurately classify iris flowers into three species, demonstrating fundamental classification techniques and achieving near-perfect accuracy.

## Dataset

**File:** `iris.csv`

**Description:** Classic Iris dataset with flower measurements for three species

**Key Features:**
- **Sepal Length (cm):** Length of the sepal
- **Sepal Width (cm):** Width of the sepal
- **Petal Length (cm):** Length of the petal
- **Petal Width (cm):** Width of the petal
- **Species (target):** Iris Setosa, Iris Versicolor, Iris Virginica

**Dataset Characteristics:**
- 150 samples (50 per species)
- 4 numerical features
- 3 balanced classes
- No missing values
- Well-separated classes

## Objectives

1. **Exploratory Data Analysis:**
   - Feature distribution analysis by species
   - Pairwise scatter plots for visual separation
   - Correlation analysis between measurements
   - Species distinguishing characteristics identification

2. **Feature Analysis:**
   - Which features best separate species?
   - Petal vs sepal measurements discriminative power
   - Linear separability assessment
   - Decision boundary visualization

3. **Classification Modeling:**
   - Compare multiple classifiers:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - Naive Bayes
   - Cross-validation for robust evaluation
   - Confusion matrix analysis

4. **Model Interpretation:**
   - Feature importance ranking
   - Decision boundary plots (2D projections)
   - Misclassification analysis
   - Model confidence assessment

## Analysis Techniques

- Multi-class classification algorithms
- Feature scaling and normalization
- Principal Component Analysis (PCA) for 2D visualization
- K-Fold cross-validation
- Grid search for hyperparameter tuning
- Decision boundary visualization
- Confusion matrix and classification metrics

## Expected Outcomes

- Classification accuracy >95% (near-perfect expected)
- Petal dimensions identified as most discriminative features
- Clear visual separation in scatter plots
- Iris Setosa perfectly separable from other species
- Some overlap between Versicolor and Virginica
- Production-ready classification pipeline
- Interactive prediction tool

## Visualizations

- Pair plots (scatter plot matrix) colored by species
- Feature distribution histograms by species
- Correlation heatmap
- 2D decision boundary plots (using PCA or feature pairs)
- Box plots comparing features across species
- Confusion matrices for each classifier
- ROC curves (one-vs-rest)
- Feature importance bar charts

## Key Insights Expected

- **Petal Length** and **Petal Width** are the most discriminative features
- Iris Setosa has distinctly smaller petals (linearly separable)
- Iris Versicolor and Virginica show some overlap
- Simple models (Logistic Regression, KNN) perform excellently
- Near-perfect classification achievable (98-100% accuracy)
- Minimal feature engineering required

## Why This Dataset Matters

- **Educational Value:** Perfect for learning ML fundamentals
- **Benchmark:** Standard for testing classification algorithms
- **Interpretability:** Simple enough to fully understand
- **Visual:** Easy to visualize in 2D/3D space
- **Well-Behaved:** Clean data with clear patterns

## Project Structure

- `data/` - Iris dataset CSV file
- `models/` - Trained classifiers and preprocessing pipelines
- `notebooks/` - EDA, classification modeling, and visualization notebooks
- `viz/` - Scatter plots, decision boundaries, species comparisons

## Getting Started

This is an excellent introductory ML project. Start with the EDA notebook to visualize species differences, then explore the classification notebook to see how various algorithms achieve near-perfect accuracy on this well-structured dataset.
