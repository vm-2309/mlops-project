import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from source.data_processing import load_and_process_data

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/crop_recommendation.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(crop_model, "models/crop_model.pkl")
joblib.dump(risk_model, "models/risk_model.pkl")

CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")

# -----------------------------
# SET MLFLOW TRACKING URI
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_and_process_data(DATA_PATH)

feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[feature_cols]

# Targets
y_crop = df["label"]
y_risk = df["risk_level"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_crop_train, y_crop_test = train_test_split(
    X, y_crop, test_size=0.2, random_state=42
)

_, _, y_risk_train, y_risk_test = train_test_split(
    X, y_risk, test_size=0.2, random_state=42
)

# -----------------------------
# MLFLOW EXPERIMENT
# -----------------------------
mlflow.set_experiment("Smart Crop Prediction")

with mlflow.start_run():

    # -----------------------------
    # CROP MODEL
    # -----------------------------
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train, y_crop_train)
    crop_preds = crop_model.predict(X_test)
    crop_acc = accuracy_score(y_crop_test, crop_preds)

    # -----------------------------
    # RISK MODEL
    # -----------------------------
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    risk_model.fit(X_train, y_risk_train)
    risk_preds = risk_model.predict(X_test)
    risk_acc = accuracy_score(y_risk_test, risk_preds)

    # -----------------------------
    # SAVE MODELS
    # -----------------------------
    joblib.dump(crop_model, CROP_MODEL_PATH)
    joblib.dump(risk_model, RISK_MODEL_PATH)

    # -----------------------------
    # LOG PARAMETERS
    # -----------------------------
    mlflow.log_param("algorithm", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # -----------------------------
    # LOG METRICS
    # -----------------------------
    mlflow.log_metric("crop_accuracy", crop_acc)
    mlflow.log_metric("risk_accuracy", risk_acc)

    # -----------------------------
    # LOG ARTIFACTS
    # -----------------------------
    mlflow.log_artifact(CROP_MODEL_PATH)
    mlflow.log_artifact(RISK_MODEL_PATH)

    # -----------------------------
    # REGISTER MODELS
    # -----------------------------
    crop_model_info = mlflow.sklearn.log_model(
        sk_model=crop_model,
        artifact_path="crop_model",
        registered_model_name="CropRecommendationModel"
    )

    risk_model_info = mlflow.sklearn.log_model(
        sk_model=risk_model,
        artifact_path="risk_model",
        registered_model_name="CropRiskModel"
    )

    print("✅ Training complete!")
    print(f"🌾 Crop Model Accuracy: {crop_acc:.4f}")
    print(f"⚠️ Risk Model Accuracy: {risk_acc:.4f}")
    print("📦 Models saved in /models")
    print("📊 MLflow run logged successfully")
    print("🧾 Models registered in MLflow Registry")