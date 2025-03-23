from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path
from .utils import calculate_metrics, plot_feature_importance

class CreditCardApprovalModel:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        
    def train(self, X_train, y_train, param_grid=None):
        """Train the model with optional hyperparameter tuning."""
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        base_model = RandomForestClassifier(random_state=42)
        self.model = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Best parameters:", self.model.best_params_)
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """Get and plot feature importance."""
        if not hasattr(self.model, 'best_estimator_'):
            raise ValueError("Model has not been trained with GridSearchCV.")
        
        importances = self.model.best_estimator_.feature_importances_
        return plot_feature_importance(feature_names, importances)
    
    def save_model(self, path=None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_path = path or self.model_path
        if save_path is None:
            raise ValueError("No path specified to save the model.")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path=None):
        """Load a trained model."""
        load_path = path or self.model_path
        if load_path is None:
            raise ValueError("No path specified to load the model.")
        
        self.model = joblib.load(load_path)
        print(f"Model loaded from {load_path}")
        return self.model