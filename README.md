# Customer-Churn-Prediction-using-Machine-Learning-in-R
This repository contains an end-to-end machine learning pipeline in R for predicting customer churn. It performs comprehensive EDA, feature engineering, statistical testing, model training, evaluation, and visualization using supervised learning techniques.

## Overview
The goal of this project is to predict whether a customer will churn using structured data. The workflow includes:
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Statistical Tests (T-test, Chi-Square, Mann-Whitney)
- Model Training with Cross-Validation
- Performance Evaluation (Accuracy, AUC, F1, Precision)
- Visualizations: Distribution plots, boxplots, ROC curves, confusion matrices

## Tools & Libraries Used
- tidyverse – data manipulation & visualization
- caret – machine learning models & preprocessing
- pROC – ROC curve & AUC
- corrplot – correlation matrix
- xgboost – gradient boosting model
- ggplot2, gridExtra, viridis – advanced plots
- knitr – markdown table rendering

## Models Used
Three models were trained and compared:
- Model	Algorithm
- Logistic Regression	
- Random Forest	
- XGBoost	
Each was evaluated using 5-fold cross-validation and tested on an unseen 20% split.

## Model Evaluation
Performance metrics evaluated:
- Accuracy
- ROC AUC
- Precision
- F1 Score
- Recall

We want to focus on False Negatives, in other words:
False Negative (FN): A customer who will churn, but the model predicts they won’t.

These are real churners that my model missed — the ones we should have acted on but didn't.
We are not concerned with customers who are falsely alarmed as churners (those are False Positives).

So the most important factor here is Recall.
Recall (a.k.a Sensitivity or True Positive Rate)
Measures how many actual churners you correctly identified:

Recall =True Positives/(True Positives + False Negatives)

- A high recall means: We are catching most of the customers who will churn.
- Low FN = Less revenue loss due to missed intervention.

## Visual Outputs
The script generates:
- Confusion Matrices
- ROC Curves
- Boxplots by Churn
- Feature Distributions
- Correlation Matrix
- Proportional Bar Plots by Categorical Features

## Preprocessing Highlights
- Binary & one-hot encoding of categorical variables
- Handling missing values in TotalCharges
- Standardization of numeric features
- Removing non-predictive IDs and redundant columns

## Final Result

After evaluating three classification models — Logistic Regression, Random Forest, and XGBoost — on multiple performance metrics, XGBoost emerges as the most suitable model for our churn prediction task.

Despite Logistic Regression having slightly higher AUC_ROC (0.8619 vs. 0.8659), XGBoost demonstrates the best overall balance:
- Highest F1 Score (0.6272), indicating strong balance between precision (0.7153) and recall (0.5584).
- Highest Accuracy (0.8345), suggesting strong overall classification.
Recall is crucial in our context, as we aim to minimize false negatives i.e., customers wrongly predicted as staying when they are likely to churn. XGBoost’s recall (0.5584) is significantly better than Random Forest (0.3732) and nearly matches Logistic Regression (0.5613), but with better overall metrics.

## Dataset

Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn





