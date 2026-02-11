"""
Day 4: Spotify Track Data Analysis
Run this script to generate models and visualizations.
"""
import pandas as pd
import numpy as np
import os
import sys
import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths (relative to day4/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'viz')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=== Loading Data ===")
track_data = pd.read_csv(os.path.join(DATA_DIR, 'track_data_final.csv'))
spotify_clean = pd.read_csv(os.path.join(DATA_DIR, 'spotify_data_clean.csv'))

print(f"track_data_final.csv: {track_data.shape}")
print(f"spotify_data_clean.csv: {spotify_clean.shape}")

# --- Data Cleaning ---
print("\n=== Data Cleaning ===")
df = track_data.dropna(subset=['artist_popularity', 'artist_followers']).copy()
df['explicit'] = df['explicit'].astype(int)
df['release_year'] = pd.to_datetime(df['album_release_date'], errors='coerce').dt.year
df['track_duration_min'] = df['track_duration_ms'] / 60000

df['popularity_category'] = pd.cut(
    df['track_popularity'],
    bins=[-1, 25, 50, 75, 100],
    labels=['Low', 'Medium', 'High', 'Very High']
)
print(f"Cleaned shape: {df.shape}")
print(f"Popularity categories:\n{df['popularity_category'].value_counts().sort_index()}")

# --- Visualizations ---
print("\n=== Generating Visualizations ===")

# 1. Popularity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['track_popularity'], bins=40, color='#1DB954', edgecolor='black', alpha=0.8)
axes[0].set_title('Distribution of Track Popularity')
axes[0].set_xlabel('Track Popularity')
axes[0].set_ylabel('Count')

cat_counts = df['popularity_category'].value_counts().sort_index()
axes[1].bar(cat_counts.index.astype(str), cat_counts.values,
            color=['#e74c3c','#f39c12','#2ecc71','#3498db'], edgecolor='black')
axes[1].set_title('Tracks by Popularity Category')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'popularity_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/popularity_distribution.png")

# 2. Artist vs Track popularity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df['artist_popularity'], df['track_popularity'], alpha=0.3, s=10, color='#1DB954')
axes[0].set_title('Artist Popularity vs Track Popularity')
axes[0].set_xlabel('Artist Popularity')
axes[0].set_ylabel('Track Popularity')

axes[1].scatter(df['artist_followers'], df['track_popularity'], alpha=0.3, s=10, color='#e74c3c')
axes[1].set_title('Artist Followers vs Track Popularity')
axes[1].set_xlabel('Artist Followers')
axes[1].set_ylabel('Track Popularity')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'artist_vs_track_popularity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/artist_vs_track_popularity.png")

# 3. Correlation heatmap
numeric_cols = ['track_popularity', 'track_duration_ms', 'artist_popularity',
                'artist_followers', 'track_number', 'album_total_tracks', 'explicit']
corr = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/correlation_heatmap.png")

# 4. Top 15 artists
top_artists = (
    df.groupby('artist_name')['track_popularity']
    .agg(['mean', 'count'])
    .query('count >= 3')
    .sort_values('mean', ascending=False)
    .head(15)
)
plt.figure(figsize=(12, 6))
plt.barh(top_artists.index[::-1], top_artists['mean'].values[::-1], color='#1DB954', edgecolor='black')
plt.xlabel('Average Track Popularity')
plt.title('Top 15 Artists by Average Track Popularity (min 3 tracks)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'top_artists.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/top_artists.png")

# 5. Box plot by album type
plt.figure(figsize=(10, 5))
album_types = df['album_type'].value_counts().index[:5]
subset = df[df['album_type'].isin(album_types)]
sns.boxplot(data=subset, x='album_type', y='track_popularity', palette='Set2')
plt.title('Track Popularity by Album Type')
plt.xlabel('Album Type')
plt.ylabel('Track Popularity')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'popularity_by_album_type.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/popularity_by_album_type.png")

# 6. Explicit vs non-explicit
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='explicit', y='track_popularity', palette=['#2ecc71', '#e74c3c'])
plt.xticks([0, 1], ['Non-Explicit', 'Explicit'])
plt.title('Track Popularity: Explicit vs Non-Explicit')
plt.xlabel('Explicit')
plt.ylabel('Track Popularity')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'explicit_vs_popularity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/explicit_vs_popularity.png")

# 7. Genre analysis
def parse_genres(genre_str):
    if pd.isna(genre_str) or genre_str in ('N/A', '[]', ''):
        return []
    try:
        return ast.literal_eval(genre_str)
    except (ValueError, SyntaxError):
        return [genre_str.strip()]

all_genres = df['artist_genres'].apply(parse_genres).explode()
genre_counts = all_genres.value_counts().head(20)

plt.figure(figsize=(12, 6))
plt.barh(genre_counts.index[::-1], genre_counts.values[::-1], color='#3498db', edgecolor='black')
plt.xlabel('Number of Tracks')
plt.title('Top 20 Genres by Track Count')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'top_genres.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/top_genres.png")

# --- Classification Models ---
print("\n=== Training Classification Models ===")

feature_cols = ['track_duration_ms', 'artist_popularity', 'artist_followers',
                'track_number', 'album_total_tracks', 'explicit']

model_df = df[feature_cols + ['popularity_category']].dropna()

le = LabelEncoder()
model_df = model_df.copy()
model_df['target'] = le.fit_transform(model_df['popularity_category'])

X = model_df[feature_cols]
y = model_df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print("\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred, target_names=le.classes_))

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# Model comparison chart
metrics = ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1 Score (weighted)']
lr_scores = [
    accuracy_score(y_test, lr_pred),
    precision_score(y_test, lr_pred, average='weighted'),
    recall_score(y_test, lr_pred, average='weighted'),
    f1_score(y_test, lr_pred, average='weighted')
]
rf_scores = [
    accuracy_score(y_test, rf_pred),
    precision_score(y_test, rf_pred, average='weighted'),
    recall_score(y_test, rf_pred, average='weighted'),
    f1_score(y_test, rf_pred, average='weighted')
]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#3498db')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='#1DB954')
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15)
ax.legend()
ax.set_ylim(0, 1)
ax.bar_label(bars1, fmt='%.3f', fontsize=8)
ax.bar_label(bars2, fmt='%.3f', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/model_comparison.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pred, name in [(axes[0], lr_pred, 'Logistic Regression'),
                        (axes[1], rf_pred, 'Random Forest')]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'Confusion Matrix: {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/confusion_matrices.png")

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='#1DB954', edgecolor='black')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: viz/feature_importance.png")

# --- Save Models ---
print("\n=== Saving Models ===")
joblib.dump(lr_model, os.path.join(MODELS_DIR, 'spotify_lr_model.joblib'))
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'spotify_rf_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'spotify_scaler.joblib'))
joblib.dump(le, os.path.join(MODELS_DIR, 'spotify_label_encoder.joblib'))

print("  spotify_lr_model.joblib")
print("  spotify_rf_model.joblib")
print("  spotify_scaler.joblib")
print("  spotify_label_encoder.joblib")

print("\n=== Done! ===")
