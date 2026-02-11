# Day 4: Spotify Track Data Analysis

**Author:** Olayinka Akanji

## Overview
This day focuses on exploratory data analysis (EDA) and classification modeling using Spotify track datasets. We analyze track popularity, artist metrics, genres, and other features to understand patterns in music data, then build classification models to predict track popularity categories.

## Dataset
- **Source:** [Spotify Global Music Dataset 2009-2025](https://www.kaggle.com/datasets/wardabilal/spotify-global-music-dataset-20092025)
- `track_data_final.csv` — 8,778 tracks with popularity, duration (ms), artist details, album info, and genres
- `spotify_data_clean.csv` — 8,582 tracks with cleaned Spotify data including album info and release dates

## Key Findings from EDA
- **Popularity Distribution**: Track popularity is right-skewed; most tracks fall in the High (51-75) category, followed by Medium (26-50)
- **Artist-Track Correlation**: Artist popularity has a moderate positive correlation with track popularity
- **Artist Followers**: High follower counts show some correlation with track popularity, but the relationship is noisy
- **Album Type**: Tracks from albums have wider popularity ranges than singles or compilations
- **Explicit Content**: Explicit and non-explicit tracks show similar popularity distributions
- **Top Genres**: Pop, hip-hop, and related genres dominate the dataset
- **Visualizations**: Histograms, scatter plots, box plots, heatmaps, and bar charts generated

## Models
- **Logistic Regression** — Accuracy: 52.5%, weighted Precision: 0.58, weighted F1: 0.43
- **Random Forest** — Accuracy: 55.7%, weighted Precision: 0.61, weighted F1: 0.49
- Evaluation: accuracy, precision, recall, F1 score, confusion matrix
- **Top features**: artist_popularity and artist_followers are the strongest predictors of track popularity
- Saved models: `spotify_lr_model.joblib`, `spotify_rf_model.joblib`

## Files
```
day4/
├── data/
│   ├── track_data_final.csv
│   └── spotify_data_clean.csv
├── notebooks/
│   ├── spotify_track_analysis.ipynb    # Main analysis notebook
│   └── run_analysis.py                 # Standalone script to regenerate outputs
├── models/
│   ├── spotify_lr_model.joblib         # Logistic Regression model
│   ├── spotify_rf_model.joblib         # Random Forest model
│   ├── spotify_scaler.joblib           # StandardScaler for LR input
│   └── spotify_label_encoder.joblib    # LabelEncoder for popularity categories
├── viz/
│   ├── popularity_distribution.png
│   ├── artist_vs_track_popularity.png
│   ├── correlation_heatmap.png
│   ├── top_artists.png
│   ├── popularity_by_album_type.png
│   ├── explicit_vs_popularity.png
│   ├── top_genres.png
│   ├── model_comparison.png
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
plotly
```

## Usage
Open `notebooks/spotify_track_analysis.ipynb` in Jupyter and run all cells, or run the standalone script:
```bash
cd day4/notebooks
python run_analysis.py
```

This will:
1. Load and clean the Spotify datasets
2. Generate EDA visualizations (saved to `viz/`)
3. Train Logistic Regression and Random Forest classifiers
4. Evaluate and compare model performance
5. Save trained models to `models/`

## Author
**Olayinka Akanji**
