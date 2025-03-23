import os
from pathlib import Path
from src.data_cleaning import load_data, clean_data
from src.preprocessing import CreditCardDataPreprocessor
from src.modeling import CreditCardApprovalModel
from src.predict import CreditCardApprovalPredictor

def main():
    # Set up paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load and clean data
    print("Loading and cleaning data...")
    raw_data = load_data(DATA_DIR / "default of credit card clients.xls")
    if raw_data is None:
        return
    
    print("\nColumns in the dataset:")
    print(raw_data.columns.tolist())
    
    cleaned_data = clean_data(raw_data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = CreditCardDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        cleaned_data,
        target_column='Y'  # Target column for default prediction
    )
    
    # Train model
    print("\nTraining model...")
    model = CreditCardApprovalModel(model_path=MODEL_DIR / "credit_card_model.joblib")
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    # Plot and save feature importance
    print("\nGenerating feature importance plot...")
    fig = model.get_feature_importance(preprocessor.feature_names)
    fig.savefig(MODEL_DIR / "feature_importance.png")
    
    # Save model
    print("\nSaving model...")
    model.save_model()
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
