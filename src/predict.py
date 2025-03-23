import pandas as pd
from .preprocessing import CreditCardDataPreprocessor

class CreditCardApprovalPredictor:
    def __init__(self, model, preprocessor):
        """
        Initialize predictor with trained model and preprocessor.
        
        Args:
            model: Trained model instance
            preprocessor: Fitted CreditCardDataPreprocessor instance
        """
        self.model = model
        self.preprocessor = preprocessor
    
    def predict(self, applicant_data):
        """
        Predict credit card approval for new applicant(s).
        
        Args:
            applicant_data: DataFrame containing applicant information
                          with same features as training data
        
        Returns:
            Dictionary containing prediction and probability
        """
        # Preprocess the applicant data
        processed_data = self.preprocessor.transform_new_data(applicant_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)[:, 1]
        
        results = []
        for pred, prob in zip(prediction, probability):
            results.append({
                'approved': bool(pred),
                'approval_probability': float(prob),
                'confidence': 'High' if abs(prob - 0.5) > 0.3 else 'Medium' if abs(prob - 0.5) > 0.15 else 'Low'
            })
        
        return results[0] if len(results) == 1 else results
    
    def explain_prediction(self, applicant_data):
        """
        Provide explanation for the prediction based on feature importance.
        
        Args:
            applicant_data: DataFrame containing single applicant information
        
        Returns:
            Dictionary containing prediction explanation
        """
        if len(applicant_data) != 1:
            raise ValueError("Explanation is only available for single applicant")
        
        # Get prediction
        prediction = self.predict(applicant_data)
        
        # Get feature importance
        feature_importance = zip(
            self.preprocessor.feature_names,
            self.model.best_estimator_.feature_importances_
        )
        
        # Sort features by importance
        top_features = sorted(
            feature_importance,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        explanation = {
            'prediction': prediction,
            'key_factors': [
                {
                    'feature': feature,
                    'importance': float(importance),
                    'value': applicant_data[feature].iloc[0]
                }
                for feature, importance in top_features
            ]
        }
        
        return explanation
