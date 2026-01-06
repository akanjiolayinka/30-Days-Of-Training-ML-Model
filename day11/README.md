# Day 11: Loan Eligibility Prediction

**Author:** Olayinka Akanji

## Overview
This project analyzes loan application data to predict loan eligibility using machine learning classification models.

## Dataset
- **Name:** Loan Eligibility Prediction Dataset
- **Size:** 614 records with 13 features
- **Target:** Loan_Status (Y/N)

## Features
- Customer_ID: Unique identifier
- Gender: Male/Female
- Married: Yes/No
- Dependents: Number of dependents
- Education: Graduate/Not Graduate
- Self_Employed: Yes/No
- Applicant_Income: Income of applicant
- Coapplicant_Income: Income of co-applicant
- Loan_Amount: Loan amount requested
- Loan_Amount_Term: Term of loan in months
- Credit_History: Credit history meets guidelines
- Property_Area: Urban/Semiurban/Rural
- Loan_Status: Loan approved (Y/N)

## Analysis Performed
1. **Data Exploration**
   - Statistical summary of features
   - Missing value analysis
   - Distribution analysis

2. **Data Visualization**
   - Categorical feature distributions
   - Gender vs Loan Status analysis
   - Credit History impact visualization

3. **Model Development**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

4. **Model Comparison**
   - Precision, Recall, and F1 Score metrics
   - Visual comparison of model performance

## Key Findings
- Male applicants submit more applications than female applicants
- Approval is strongly associated with credit history
- Married applicants have higher approval rates
- Applicants from semiurban areas receive approvals more frequently
- Credit score and verified income are primary determinants of loan approval

## Models Performance
- **Logistic Regression:** High precision and recall
- **Decision Tree:** Good performance with depth=4
- **Random Forest:** Best overall performance (selected as final model)

## Files
- `data/Loan_Eligibility_Prediction.csv`: Raw dataset
- `notebooks/loan_eligibility_prediction.ipynb`: Analysis notebook
- `models/loan_eligibility_rf.joblib`: Trained Random Forest model

## Requirements
```python
pandas
numpy
plotly
scikit-learn
joblib
```

## Usage
Open the Jupyter notebook and run all cells to:
1. Load and explore the data
2. Visualize feature relationships
3. Train multiple classification models
4. Compare model performance
5. Save the best model for predictions

## Author
**Olayinka Akanji**
