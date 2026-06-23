# Telco Customer Churn - Machine Learning Experimentation Pipeline

## Description
An end-to-end data science experimentation workflow built to predict subscriber churn probabilities using advanced classification techniques. Integrated headlessly with DagsHub MLflow remote server for tracking parameters and metrics.

## Dataset
Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) with full feature engineering (handling blanks in `TotalCharges`, dummy encoding, categorical mapping).

## MLflow Architecture
Explicit autologging configuration logging parameters, execution metrics (Accuracy, Precision, Recall, F1-Score), and registering the optimized Random Forest champion model under the production target registry string `"credit-scoring-model"`.

## How to Run
Explicit instructions on running `Eksperimen_Lukman.ipynb` locally or via Google Colab:
1. Ensure your remote OAuth headless token variables are set in your environment: `DAGSHUB_CLIENT_TOKEN` or `MLFLOW_TRACKING_PASSWORD`.
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells sequentially. The dataset will be preprocessed, the model will train, and metrics will automatically sync to the DagsHub remote tracking server.
