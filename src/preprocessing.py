import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class CreditCardDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess_data(self, df, target_column, test_size=0.2, random_state=42):
        """Preprocess the credit card data for modeling."""
        # Create a copy
        df_processed = df.copy()
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Process categorical and numerical columns separately
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(exclude=[np.number]).columns
        
        # Label encode categorical columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Scale numerical columns
        for col in numeric_columns:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessors."""
        if self.feature_names is None:
            raise ValueError("Preprocessor has not been fitted. Run preprocess_data first.")
        
        df_processed = df.copy()
        
        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            df_processed[col] = encoder.transform(df_processed[col])
        
        # Apply scaling
        for col, scaler in self.scalers.items():
            df_processed[col] = scaler.transform(df_processed[col].values.reshape(-1, 1))
        
        return df_processed[self.feature_names]
