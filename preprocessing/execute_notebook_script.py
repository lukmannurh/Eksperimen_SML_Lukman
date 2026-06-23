import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read raw_data.csv into a pandas DataFrame
data_path = 'raw_data.csv'
df = pd.read_csv(data_path)

# Display basic information
print("--- Data Info ---")
df.info()

print("\n--- Data Description ---")
print(df.describe(include='all'))

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Display distribution of the target column
print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts())

sns.countplot(data=df, x='Churn')
plt.title('Distribution of Target Column (Churn)')
plt.close("all")
# Drop the customerID column
df = df.drop(columns=['customerID'])

# Crucial Cleaning: Convert TotalCharges to numeric, coerce errors, and fill NaNs with median
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)

# Convert target column Churn to binary integers (1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Cleanly encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Perform an 80:20 Train-Test split
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save fully processed, numeric-only training features and labels together
out_dir = r'D:\dicoding\SMSML_Lukman\Membangun_model\namadataset_preprocessing'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'cleaned_data.csv')

cleaned_df = pd.concat([X, y], axis=1)
cleaned_df.to_csv(out_path, index=False)
print(f"Fully processed numeric-only data saved to: {out_path}")
# 1. Jalankan instalasi library pendukung

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. ATUR AUTENTIKASI SECARA HEADLESS (Bypass dagshub.init)
# Tempel token DagsHub kamu di bawah ini
DAGSHUB_TOKEN = "618385ed8cabd61fa55393eeb632a3021b166ff4"

os.environ["MLFLOW_TRACKING_USERNAME"] = "lukmannurh"
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri("https://dagshub.com/lukmannurh/Eksperimen_SML_Lukman.mlflow")

# Aktifkan fitur autologging dari MLflow
mlflow.sklearn.autolog(log_models=True)

# 3. MEMASTIKAN DATA SUDAH TERLOAD (Jaga-jaga jika runtime ter-reset)
# Pastikan file raw_data.csv sudah di-upload ke folder Colab
if os.path.exists('/content/raw_data.csv'):
    df_raw = pd.read_csv('/content/raw_data.csv')
    # Preprocessing kilat agar X_train dan y_train terdefinisi di cell ini
    df_clean = df_raw.drop(columns=['customerID'], errors='ignore').copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    df_clean = pd.get_dummies(df_clean, drop_first=True)

    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset Telco Churn berhasil dimuat dan dipisahkan!")
else:
    print("⚠️ PERINGATAN: File 'raw_data.csv' belum di-upload ke sidebar kiri Google Colab!")

# 4. MEMULAI PROSES TRAINING & LOGGING KE CLOUD DAGSHUB
print("\nStarting MLflow experiment tracking session...")
with mlflow.start_run(run_name="Telco_Churn_Optimization"):

    # Inisialisasi model baseline
    rf = RandomForestClassifier(random_state=42)

    # Atur strategi hyperparameter tuning
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Proses pelatihan model
    search.fit(X_train, y_train)

    # Ambil model terbaik (Champion Model)
    champion_model = search.best_estimator_
    print("\nBest parameters found:", search.best_params_)

    # Evaluasi hasil prediksi
    y_pred = champion_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Registrasikan model final ke dalam MLflow Model Registry Dicoding target
    mlflow.sklearn.log_model(
        sk_model=champion_model,
        artifact_path="champion_model_artifacts",
        registered_model_name="credit-scoring-model"
    )
    print("\n[SUCCESS] Champion model baru resmi terdaftar di Model Registry DagsHub!")
from sklearn.metrics import classification_report, confusion_matrix

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn (0)', 'Churn (1)'],
            yticklabels=['No Churn (0)', 'Churn (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.close("all")
