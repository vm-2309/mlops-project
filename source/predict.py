import os
import csv
import joblib
from datetime import datetime
import pandas as pd

# -----------------------------
# PATHS
# -----------------------------
CROP_MODEL_PATH = "models/crop_model.pkl"
RISK_MODEL_PATH = "models/risk_model.pkl"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_history.csv")

# Ensure logs folder exists
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
crop_model = joblib.load(CROP_MODEL_PATH)
risk_model = joblib.load(RISK_MODEL_PATH)


# -----------------------------
# ADVISORY LOGIC
# -----------------------------
def generate_advisory(ph, rainfall):
    if rainfall > 200:
        return "Suitable for high rainfall crops. Ensure proper drainage."
    elif rainfall < 50:
        return "Low rainfall detected. Irrigation is recommended."
    elif ph < 5.8:
        return "Soil is acidic. Consider pH balancing before cultivation."
    elif ph > 7.8:
        return "Soil is alkaline. Soil treatment may improve productivity."
    else:
        return "Conditions are generally suitable for cultivation."


# -----------------------------
# LOG PREDICTION
# -----------------------------
def log_prediction(data, crop_prediction, risk_prediction, advisory):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write header only if file is new
        if not file_exists:
            writer.writerow([
                "timestamp",
                "N", "P", "K",
                "temperature", "humidity", "ph", "rainfall",
                "predicted_crop", "predicted_risk", "advisory"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"], data["ph"], data["rainfall"],
            crop_prediction, risk_prediction, advisory
        ])


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def predict_crop_and_risk(input_data):
    """
    input_data = {
        "N": ...,
        "P": ...,
        "K": ...,
        "temperature": ...,
        "humidity": ...,
        "ph": ...,
        "rainfall": ...
    }
    """

    df = pd.DataFrame([input_data])

    crop_prediction = crop_model.predict(df)[0]
    risk_prediction = risk_model.predict(df)[0]
    advisory = generate_advisory(input_data["ph"], input_data["rainfall"])

    # Log prediction for monitoring
    log_prediction(input_data, crop_prediction, risk_prediction, advisory)

    return {
        "recommended_crop": crop_prediction,
        "risk_level": risk_prediction,
        "advisory": advisory
    }