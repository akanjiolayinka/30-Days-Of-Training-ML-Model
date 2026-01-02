# Day 26: Goodreads Books Analysis & Rating Prediction

## Project Overview

This project analyzes book data from Goodreads to uncover reading trends, author patterns, and factors influencing book ratings. By examining book metadata, user ratings, publication details, and genre information, we build models to predict book popularity and provide insights for readers, authors, and publishers.

## Dataset

**File:** `goodreads_books_dataset.csv`

**Description:** Comprehensive Goodreads book records with ratings, metadata, and engagement metrics

**Key Features:**
- **Book Information:** Title, author, ISBN, publication year
- **Ratings:** Average rating, number of ratings, rating distribution
- **Engagement:** Number of reviews, ratings count, text reviews count
- **Metadata:** Genre/tags, page count, language, series information
- **Publisher:** Publisher name, publication date, edition
- **Popularity Metrics:** Shelves added to, want-to-read count

## Objectives

1. **Book Popularity Analysis:**
   - Distribution of ratings across books
   - Most highly-rated books and authors
   - Relationship between ratings count and average rating
   - Publication year trends in ratings

2. **Author & Publisher Insights:**
   - Top authors by average rating and popularity
   - Prolific authors vs quality assessment
   - Publisher reputation analysis
   - Series vs standalone book performance

3. **Text & Title Analysis:**
   - Title length impact on ratings
   - Title complexity and readability scores
   - Genre/tag frequency and popularity
   - Book description sentiment analysis

4. **Predictive Modeling:**
   - Rating prediction based on book attributes:
     - Random Forest Regressor
     - Gradient Boosting (XGBoost, AdaBoost)
     - Text-based features (title, description)
   - Popularity classification (high/medium/low)
   - Feature importance for ratings

## Analysis Techniques

- Text feature engineering (title length, word count, sentiment)
- Natural Language Processing (NLP) for descriptions
- Regression and classification modeling
- Author encoding and aggregation
- Time series analysis for publication trends
- Clustering for genre/book similarity
- Recommendation system fundamentals

## Expected Outcomes

- Predictive model with reasonable accuracy for ratings
- Title complexity correlation with reader engagement
- Series books show distinct rating patterns
- Author information highly predictive of ratings
- Publication year trends revealing reading preferences shift
- Genre popularity rankings
- Optimal book length for highest ratings
- Interactive book recommendation insights

## Visualizations

- Rating distribution histograms
- Top authors and books bar charts
- Scatter plots (Ratings Count vs Average Rating)
- Publication year trend lines
- Genre/tag word clouds
- Author performance heatmaps
- Title length vs rating scatter plots
- Feature importance rankings
- Correlation matrix of numerical features

## Key Insights Expected

- Title complexity influences reader perception
- Series books have distinct rating patterns vs standalone
- Author reputation is highly predictive
- Page count optimal range for highest ratings
- Publication recency bias in ratings
- Genre-specific rating distributions
- Highly-rated books don't always have most ratings

## Text Analysis Features

- **Title Analysis:**
  - Word count in title
  - Character count
  - Readability scores
  - Presence of series indicators
  
- **Description Analysis:**
  - Sentiment polarity
  - Description length
  - Key theme extraction

## Project Structure

- `data/` - Goodreads books dataset
- `models/` - Rating prediction models, text vectorizers, pipelines
- `notebooks/` - EDA, text analysis, and predictive modeling notebooks
- `viz/` - Book trends, author analysis, rating distributions, genre insights

## Getting Started

Begin with the EDA notebook to explore book ratings distributions and popular authors, then review the text analysis notebook for title and description insights, and finally explore the modeling notebook for rating prediction.
