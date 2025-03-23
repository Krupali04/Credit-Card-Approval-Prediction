# Credit Card Default Prediction System

This project implements a machine learning system for predicting credit card defaults using historical credit card data. The system uses a Random Forest classifier to identify patterns and predict whether a credit card holder is likely to default on their payments.

## Project Overview

Credit card default prediction is a critical task for financial institutions to manage risk and make informed lending decisions. This system:

- Processes and cleans credit card transaction and payment data
- Engineers relevant features for default prediction
- Trains a Random Forest model with hyperparameter tuning
- Achieves 82.02% accuracy in predicting defaults
- Provides probability scores and confidence levels for predictions

## Model Performance

The current model achieves the following metrics:
- Accuracy: 82.02%
- Precision: 67.55%
- Recall: 34.93%
- F1 Score: 46.05%
- ROC AUC: 77.29%

## Features

- Data preprocessing and cleaning pipeline
- Feature engineering and standardization
- Model training with cross-validation
- Hyperparameter optimization using GridSearchCV
- Feature importance analysis
- Prediction explanations for new applicants
