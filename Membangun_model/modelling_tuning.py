import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import dagshub
import mlflow
import mlflow.sklearn

def main():
    """
    Advanced Tier Requirement:
    Hyperparameter tuning with manual MLflow tracking mapped remotely to DagsHub.
    """
    # 1. Integrate remote tracking to DagsHub online repository
    dagshub.init(repo_owner='lukmannurh', repo_name='Eksperimen_SML_Lukman', mlflow=True)
    
    # Set the experiment name for remote tracking
    mlflow.set_experiment("Latihan Credit Scoring")
    
    # Load the preprocessed dataset
    data_path = os.path.join("preprocessing", "namadataset_preprocessing", "cleaned_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Ensure the preprocessing script has been run.")
        
    df = pd.read_csv(data_path)
    
    target_col = 'category' if 'category' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Implement Hyperparameter Tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    # 3. DO NOT use autolog. Strictly Manual Logging inside the tuning flow.
    with mlflow.start_run(run_name="Tuned RandomForest") as run:
        print("Running GridSearchCV for Hyperparameter Tuning...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # 4. Log best parameters manually
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)
            
        print(f"Best Parameters logged: {grid_search.best_params_}")
            
        # Predict on test set
        y_pred = best_model.predict(X_test)
        
        # Calculate performance metrics (using macro average for generalized multi-class support)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # 5. Log performance metrics manually
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Metrics logged: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}")
        
        # 6. Generate and manually log custom artifacts
        
        # Custom Artifact 1: metric_info.json
        metric_info = {
            "evaluation_metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            },
            "best_hyperparameters": grid_search.best_params_
        }
        
        # MLflow native log_dict securely saves dictionaries as JSON artifacts
        mlflow.log_dict(metric_info, "metric_info.json")
        print("Custom Artifact Logged: metric_info.json")
        
        # Custom Artifact 2: training_confusion_matrix.png
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Training Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        cm_path = "training_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Manually log the image artifact
        mlflow.log_artifact(cm_path)
        os.remove(cm_path) # Clean up local image after logging
        print("Custom Artifact Logged: training_confusion_matrix.png")
        
        # 7. Register the final best-tuned model into the MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="credit-scoring-model"
        )
        print("Model securely registered as 'credit-scoring-model' in MLflow Model Registry.")
        print(f"Tuning run complete. DagsHub MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
