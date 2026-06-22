import os
import argparse
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_data(file_path):
    """Safely ingest the raw dataset."""
    logging.info(f"Loading data from {file_path}...")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        logging.info(f"Successfully loaded data with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Handle missing values, duplicates, and type casting based on standard EDA practices."""
    logging.info("Starting data cleaning...")
    try:
        # Drop duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        if initial_shape != df.shape:
            logging.info(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
            
        # Handle missing values (Example: fill with median/mode)
        # Drop columns with more than 50% missing values
        missing_threshold = 0.5
        cols_to_drop = df.columns[df.isnull().mean() > missing_threshold]
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
            logging.info(f"Dropped columns due to high missing values (>50%): {list(cols_to_drop)}")
            
        # Fill remaining missing values
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        logging.info("Data cleaning completed.")
        return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise

def feature_engineering(df):
    """Perform scaling, encoding, or feature transformations."""
    logging.info("Starting feature engineering...")
    try:
        # Separate columns by type
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Scaling numeric columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logging.info(f"Scaled numeric features: {list(numeric_cols)}")
            
        # Encoding categorical columns
        if len(categorical_cols) > 0:
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col].astype(str))
            logging.info(f"Encoded categorical features: {list(categorical_cols)}")
            
        logging.info("Feature engineering completed.")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

def save_preprocessed_data(df, output_path):
    """Save the engineered data into the target directory."""
    logging.info(f"Saving preprocessed data to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info("Preprocessed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to the raw dataset (CSV/Excel)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the preprocessed dataset (CSV)")
    
    args = parser.parse_args()
    
    try:
        # Pipeline execution
        df = load_data(args.input)
        df_clean = clean_data(df)
        df_engineered = feature_engineering(df_clean)
        save_preprocessed_data(df_engineered, args.output)
        
        logging.info("Preprocessing pipeline finished successfully.")
        
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
