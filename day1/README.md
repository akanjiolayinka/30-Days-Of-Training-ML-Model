# Day 1: Netflix EDA - Content Strategy Analysis

## Dataset
**Netflix Titles Dataset** - Comprehensive catalog of Movies and TV Shows available on Netflix

Source: Netflix content library (8,800+ titles)

## Project Overview
Comprehensive exploratory data analysis of Netflix's content library trying to find unique insights from the data dump. This analysis uncovers Netflix's content strategy, naming patterns, geographic diversity, and creative partnerships.

## Data Cleaning
- Handled missing values in directors, cast, countries, ratings
- Extracted temporal features (year_added, month_added) from date_added
- Parsed duration into numeric values and units (minutes vs seasons)
- Created age_category from rating codes
- Generated genre, cast, director, and country lists for deeper analysis
- Optimized data types for memory efficiency
- **Final dataset**: Enhanced with 9+ engineered features

## Key Visualizations

### 1. Title Strategy Analysis
**Title Length and Naming Patterns**
- Title length distribution
- Average title word count by content type
- Shows Netflix favors concise titles (30-40 characters average)

### 2. Description Marketing Strategy
**Description Length by Rating**
- Box plot comparing description length across content ratings
- Reveals mature content gets longer, more detailed descriptions
- Indicates targeted audience engagement by age group

### 3. Director Concentration
**Top Netflix Directors**
- Horizontal bar chart of most prolific directors
- Identifies key creative partnerships
- Shows director concentration on Netflix platform

### 4. Geographic Production Hubs
**Content by Country**
- Top 20 countries producing Netflix content
- Reveals global production strategy
- Shows dominance of US, India, UK markets

## Unique Insights Discovered

### 1. Concise Naming Convention
Netflix uses short, punchy titles averaging 30-40 characters with mostly multi-word titles. Single-word titles represent a specific percentage of the catalog, showing deliberate naming strategy for discoverability and memorability.

### 2. Description-Based Segmentation
Mature-rated content receives 40-50% longer descriptions, targeting different audience engagement levels. This strategic approach provides more context for adult content while keeping family-friendly descriptions concise.

### 3. Strategic Director Partnerships
Small number of prolific directors dominate Netflix, suggesting curated relationships over broad creator diversity. Top directors have multiple titles, indicating ongoing partnerships rather than one-off collaborations.

### 4. Global Production Expansion
While US dominates (35-40% of content), Netflix heavily invests in international production hubs, particularly India and UK. This reflects Netflix's global expansion strategy and local content creation.

## Statistics Summary
- **Total Titles**: 8,807
- **Movies**: ~6,000 | **TV Shows**: ~2,800
- **Countries Represented**: 120+
- **Unique Directors**: 4,000+
- **Content Age Range**: 1920s to 2021
- **Average Title Length**: ~17-18 characters
- **Multi-Genre Percentage**: Majority of content spans multiple genres

## Files Structure
```
day1/
├── data/
│   ├── netflix_titles.csv           # Original dataset
│   └── cleaned/
│       └── netflix_cleaned_TADS.csv # Cleaned and enhanced dataset
├── notebooks/
│   └── netflix_eda.ipynb           # Main EDA notebook with all analysis
├── viz/
│   ├── title_strategy_analysis.html    # Interactive title analysis
│   ├── description_strategy.html       # Description length patterns
│   ├── top_directors.html              # Director concentration
│   └── country_production.html         # Geographic distribution
├── summary/
└── README.md
```

## Key Findings from Analysis

### Content Distribution
- Movies dominate the platform (~70%)
- TV Shows represent ~30% of content
- TV-MA is the most common rating
- Most content added between 2017-2021

### Rating Patterns
- **TV-MA**: Most common rating (mature audiences)
- **TV-14**: Second most common (teen audiences)
- **R**: Dominant movie rating
- **PG-13**: Popular for family-friendly movies

### Temporal Trends
- **Content Addition Peak**: 2017-2020
- **Earliest Content**: From 1920s cinema
- **Latest Content**: 2021 releases
- **Average Catalog Age**: Several years between release and addition

### Duration Analysis
- **Movies**: 90-120 minutes average
- **TV Shows**: 1-2 seasons most common
- **Shortest Movie**: 3 minutes
- **Longest Movie**: 312 minutes
- **Most Seasons**: 17 seasons

### Geographic Diversity
- **United States**: Leads production (35-40%)
- **India**: Second largest producer
- **United Kingdom**: Third largest
- **126+ countries** represented in catalog

## Technologies Used
- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **Plotly** - Interactive visualizations (Plotly Express, Graph Objects, Subplots)
- **Jupyter Notebook** - Analysis environment

## How to Run
1. Navigate to the `notebooks/` directory
2. Open `netflix_eda.ipynb` in Jupyter Notebook
3. Run all cells sequentially to reproduce the analysis
4. Interactive visualizations will be saved to the `viz/` directory
5. Cleaned data will be saved to `data/cleaned/`

## Summary of Unexpected Insights

Based on this analysis, Netflix's strategy is more nuanced than traditional content cataloging:

1. **Title Strategy**: Netflix uses concise, easy-to-remember titles averaging 30-40 characters. Single-word titles are rare (less than 20%), showing preference for descriptive multi-word titles.

2. **Description Depth**: Mature-rated content receives significantly longer, more detailed descriptions, suggesting targeted marketing for different audience engagement levels.

3. **Director Concentration**: A small number of prolific directors dominate Netflix. This suggests strategic partnerships rather than broad creator diversity—focusing on proven talent that can deliver consistent content.

4. **Geographic Localization**: The US dominates but increasingly Netflix invests in international production, particularly India and UK, reflecting a global-first content strategy.

---
*Analysis completed as part of 30 Days of Datasets*
