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

## Project Structure

```
CreditCardApprovalProject/
│
├── data/
│   └── credit_card_data.xls  # Dataset file (from UCI)
├── src/
│   ├── data_cleaning.py      # Data loading and cleaning
│   ├── preprocessing.py      # Feature engineering and preprocessing
│   ├── modeling.py           # Model training and evaluation
│   ├── predict.py           # Prediction function for new applicants
│   └── utils.py             # Helper functions
├── main.py                  # Main script to run the pipeline
└── requirements.txt         # Dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - Place the credit card dataset (`default of credit card clients.xls`) in the `data/` directory
   - The dataset should contain credit card holders' information and payment history

2. Run the Analysis Pipeline:
```bash
python main.py
```

This will:
- Clean and preprocess the data
- Train the Random Forest model
- Generate performance metrics
- Save the trained model in `models/credit_card_model.joblib`
- Create feature importance visualization in `models/feature_importance.png`

3. Making New Predictions:
   - Use the `CreditCardApprovalPredictor` class in `src/predict.py`
   - The predictor provides both predictions and confidence scores
   - Example usage can be found in the docstrings

## Data Description

The dataset contains the following information:
- X1-X23: Various features including payment history, bill amounts, and demographic information
- Y: Default payment (1 = default, 0 = non-default)

Source: UCI Machine Learning Repository - Default of Credit Card Clients Dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
