import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from source.data_processing import load_and_process_data

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Load and process dataset
df = load_and_process_data("data/crop_recommendation.csv")

# Features (inputs)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Targets (outputs)
y_crop = df['label']
y_risk = df['risk_level']

# Split data for crop model
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X, y_crop, test_size=0.2, random_state=42
)

# Split data for risk model
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X, y_risk, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("Smart Crop Recommendation MLOps")

with mlflow.start_run():

    # Train crop model
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train_crop, y_train_crop)
    crop_preds = crop_model.predict(X_test_crop)
    crop_acc = accuracy_score(y_test_crop, crop_preds)

    # Train risk model
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    risk_model.fit(X_train_risk, y_train_risk)
    risk_preds = risk_model.predict(X_test_risk)
    risk_acc = accuracy_score(y_test_risk, risk_preds)

    # Save trained models
    joblib.dump(crop_model, "models/crop_model.pkl")
    joblib.dump(risk_model, "models/risk_model.pkl")

    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    # Log accuracy metrics
    mlflow.log_metric("crop_accuracy", crop_acc)
    mlflow.log_metric("risk_accuracy", risk_acc)

    # Log models into MLflow
    mlflow.sklearn.log_model(crop_model, "crop_model")
    mlflow.sklearn.log_model(risk_model, "risk_model")

    print("Crop Model Accuracy:", round(crop_acc, 4))
    print("Risk Model Accuracy:", round(risk_acc, 4))
    print("Models saved successfully.")