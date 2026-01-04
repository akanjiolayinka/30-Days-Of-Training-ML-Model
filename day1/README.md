# Day 1: Netflix Content Analysis & Recommendation Insights

## Project Overview

This project analyzes Netflix's extensive content library of movies and TV shows to uncover content strategy patterns, genre trends, release dynamics, and global production insights. Using data from 8,800+ titles, we explore viewer preferences, content characteristics, and build foundations for recommendation systems.

## Dataset

**File:** `netflix_titles.csv`

**Description:** Comprehensive Netflix catalog with 8,810 titles (movies and TV shows)

**Key Features:**

**Content Identification:**
- Show ID (unique identifier)
- Type (Movie or TV Show)
- Title

**Production Details:**
- Director(s)
- Cast members
- Country of production
- Release year

**Platform Information:**
- Date added to Netflix
- Rating (PG, R, TV-MA, etc.)
- Duration (minutes for movies, seasons for TV shows)

**Content Categorization:**
- Listed in (genres/categories)
- Description (plot summary)

## Objectives

1. **Content Library Analysis:**
   - Movies vs TV Shows distribution
   - Content addition trends over years
   - Release year patterns and catalog age
   - Most common ratings (PG-13, R, TV-MA, etc.)

2. **Genre & Category Insights:**
   - Most popular genres on Netflix
   - Genre combinations and clustering
   - International vs US content genres
   - Documentary, drama, comedy prevalence

3. **Production Analysis:**
   - Top content-producing countries
   - Director and cast network analysis
   - International content growth trends
   - US vs international content ratios

4. **Content Characteristics:**
   - Movie duration distribution and trends
   - TV show season counts
   - Title length and complexity analysis
   - Description length vs content type

5. **Text Analytics:**
   - Genre text mining and associations
   - Description sentiment analysis
   - Title pattern recognition
   - Cast popularity analysis

## Analysis Techniques

- Text mining and natural language processing (NLP)
- Time series analysis for content addition trends
- Genre clustering and association rules
- Network analysis for directors and cast
- Statistical analysis of content attributes
- Visualization of global content distribution
- Classification models for content type prediction

## Expected Outcomes

- **Content Split:** ~70% Movies, ~30% TV Shows
- **Content Addition Peak:** 2017-2020 (pre-pandemic)
- **Top Producing Countries:** US, India, UK, Canada
- **Popular Genres:** International Movies, Dramas, Comedies, Documentaries
- **Rating Distribution:** TV-MA and R most common for mature audiences
- **Movie Duration:** 90-120 minutes average
- **TV Show Seasons:** 1-2 seasons most common
- **International Growth:** Significant increase in non-US content
- **Title Trends:** Shorter, catchier titles perform better

## Visualizations

- **Temporal:**
  - Content addition timeline (2008-2021)
  - Release year distribution
  - Seasonal addition patterns

- **Geographic:**
  - World map of content production by country
  - Top 10 producing countries bar chart
  - Regional content type preferences

- **Content Type:**
  - Movies vs TV Shows pie chart
  - Duration distribution histograms
  - Seasons count distribution for TV shows

- **Categories:**
  - Genre word cloud
  - Genre co-occurrence network
  - Rating distribution bar chart
  - Top directors and actors bar charts

- **Text Analysis:**
  - Description word clouds by genre
  - Title length distribution
  - Most common words in titles

## Key Insights Expected

### Content Strategy:
- **Shift to Originals:** Increase in Netflix Original content post-2016
- **International Expansion:** Growing library from India, South Korea, Japan
- **Mature Content:** Heavy focus on TV-MA and R-rated content
- **Binge-worthy:** Preference for 1-2 season shows (easy completion)

### Genre Trends:
- **International Movies:** Largest category
- **Documentaries:** Significant growth
- **Stand-up Comedy:** Strong representation
- **True Crime:** Popular sub-genre
- **Anime:** Growing category

### Production Insights:
- **US Dominance:** Still largest producer but declining percentage
- **India:** Second largest content provider
- **UK Productions:** High-quality dramas and comedies
- **South Korea:** K-dramas and variety shows rising
- **Spain:** Money Heist effect on Spanish content

### Content Characteristics:
- **Movie Sweet Spot:** 90-110 minutes
- **TV Shows:** 1 season (limited series trend)
- **Titles:** 2-5 words optimal
- **Descriptions:** 100-200 characters concise summaries
- **Cast Size:** Ensemble casts for TV, smaller for movies

## Feature Engineering

- **Temporal:**
  - Year added, Month added, Days on platform
  - Release year vs added year (catalog age)
  - Time lag between release and addition

- **Text Features:**
  - Title word count, character count
  - Description length, sentiment score
  - Genre count per title
  - Cast size, director count

- **Categorical:**
  - Content type (binary: Movie/TV Show)
  - Rating categories (family, teen, mature)
  - Country groups (US, International)
  - Decade of release

- **Derived:**
  - Is_Original (based on production and addition date)
  - Genre_Primary (first listed genre)
  - Multi_Country (produced in multiple countries)
  - Has_Description (non-null check)

## Machine Learning Applications

### Classification:
- **Content Type:** Predict Movie vs TV Show from description
- **Rating:** Predict content rating from genre and description
- **Genre:** Multi-label classification of genres

### Clustering:
- **Content Similarity:** Group similar shows for recommendations
- **Genre Clustering:** Find natural genre groupings
- **Country Clusters:** Similar production patterns

### NLP Tasks:
- **Genre Prediction:** From description text
- **Sentiment Analysis:** Description tone
- **Title Generation:** Learn title patterns

### Recommendation System Foundations:
- **Content-based:** Similar genres, cast, directors
- **Collaborative Filtering:** (Would need user data)
- **Hybrid Approach:** Combine content features

## Machine Learning Pipeline

```python
# Netflix Content Analysis Pipeline
1. Data Loading & Cleaning → 2. Date Parsing & Feature Engineering
→ 3. Text Processing (Tokenization, Cleaning) → 4. NLP Features
→ 5. Exploratory Analysis → 6. Visualization Creation
→ 7. Classification Models (Optional) → 8. Recommendation Prototype
```

## Content Recommendation Prototype

**Approach: Content-Based Filtering**
- Input: User's watched title
- Features: Genre, Cast, Director, Description similarity
- Output: Top 10 similar recommendations
- Method: Cosine similarity on TF-IDF vectors

## Practical Applications

- **Content Strategy:** Inform acquisition decisions
- **Marketing:** Target audiences by genre preferences
- **Production Planning:** Identify content gaps
- **Recommendation Engine:** Improve user experience
- **Trend Forecasting:** Predict future content needs
- **Competitive Analysis:** Benchmark against other platforms

## Project Structure

- `data/` - Netflix titles dataset (8,810 records)
- `models/` - Text vectorizers, classification models, recommendation engines
- `notebooks/` - EDA, text analysis, visualization, and recommendation prototypes
- `viz/` - Content trends, geographic maps, genre networks, temporal charts

## Getting Started

Begin with the EDA notebook to explore content distribution and trends, then dive into text analysis for genre and description insights, and finally build a content-based recommendation prototype to suggest similar titles.
