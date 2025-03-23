import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path):
    """Load credit card data from Excel file."""
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Clean the credit card dataset."""
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop('Unnamed: 0', axis=1)
    
    # Convert all X columns to numeric, forcing errors to NaN
    x_columns = [col for col in df_clean.columns if col.startswith('X')]
    for col in x_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert Y to numeric
    df_clean['Y'] = pd.to_numeric(df_clean['Y'], errors='coerce')
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Cleaned data shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved cleaned data to {output_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {str(e)}")

if __name__ == "__main__":
    # Test the functions
    data_path = "../data/credit_card_data.xls"
    df = load_data(data_path)
    if df is not None:
        df_clean = clean_data(df)
        save_cleaned_data(df_clean, "../data/cleaned_credit_card_data.csv")