# Day 2: Earthquake & Tsunami Risk Assessment

## Project Overview

This project analyzes global seismic activity data to understand earthquake patterns and predict tsunami occurrence risk. Using historical earthquake data including magnitude, depth, location coordinates, and tsunami occurrence, we build predictive models to identify high-risk seismic events.

## Dataset

**File:** `earthquake_data_tsunami.csv`

**Description:** Comprehensive global earthquake records with tsunami indicators

**Key Features:**
- Earthquake magnitude and intensity measurements
- Depth and geographic coordinates
- Temporal patterns and frequency
- Tsunami occurrence labels (binary classification)
- Regional seismic activity zones

## Objectives

1. **Exploratory Data Analysis:**
   - Analyze distribution of earthquake magnitudes globally
   - Identify geographic hotspots and seismic zones
   - Examine relationship between depth and magnitude
   - Temporal trend analysis of seismic activity

2. **Risk Assessment:**
   - Determine factors most correlated with tsunami occurrence
   - Analyze shallow vs deep earthquakes impact
   - Regional risk profiling

3. **Predictive Modeling:**
   - Build classification model to predict tsunami risk
   - Feature importance analysis
   - Model evaluation and performance metrics

## Analysis Techniques

- Geospatial visualization and mapping
- Statistical correlation analysis
- Machine Learning classification (Random Forest, Logistic Regression)
- Time series pattern recognition
- Risk scoring and probability estimation

## Expected Outcomes

- Interactive maps showing global seismic activity
- Classification model with 80-90% accuracy
- Identification of key tsunami risk indicators
- Regional risk assessment reports
- Insights into magnitude-depth-tsunami relationships

## Project Structure

- `data/` - Raw earthquake and tsunami datasets
- `models/` - Trained classification models and pipelines
- `notebooks/` - EDA and modeling notebooks
- `viz/` - Geographic maps, distribution plots, correlation heatmaps

## Getting Started

Explore the Jupyter notebooks in the `notebooks/` folder to see the complete analysis pipeline from data exploration to model deployment.
