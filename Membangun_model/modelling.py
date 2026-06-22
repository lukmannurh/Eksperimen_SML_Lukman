import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import os

def main():
    """
    Basic Tier Requirement: 
    Trains a baseline model with MLflow automatic logging on localhost.
    """
    # 1. Configure local MLflow tracking URI to localhost
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    # 2. Define an experiment named "Latihan Credit Scoring"
    mlflow.set_experiment("Latihan Credit Scoring")
    
    # 3. Enable automatic logging before the training phase
    mlflow.autolog()
    
    print("Starting Baseline Training with autolog...")
    
    # Load the preprocessed dataset
    data_path = os.path.join("preprocessing", "namadataset_preprocessing", "cleaned_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Ensure the preprocessing script has been run.")
        
    df = pd.read_csv(data_path)
    
    # Use the last column as the target variable for generic handling
    target_col = 'category' if 'category' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run to log baseline training
    with mlflow.start_run(run_name="Baseline RandomForest"):
        # Train a baseline Scikit-Learn classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Baseline Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
    print("Baseline model trained and logged automatically via MLflow.")

if __name__ == "__main__":
    main()
