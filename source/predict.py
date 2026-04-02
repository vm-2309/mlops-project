import joblib
import os
import pandas as pd

CROP_MODEL_PATH = "models/crop_model.pkl"
RISK_MODEL_PATH = "models/risk_model.pkl"


def load_models():
    """
    Load trained crop and risk models only when needed.
    """
    if not os.path.exists(CROP_MODEL_PATH):
        raise FileNotFoundError(f"{CROP_MODEL_PATH} not found. Please train the model first.")

    if not os.path.exists(RISK_MODEL_PATH):
        raise FileNotFoundError(f"{RISK_MODEL_PATH} not found. Please train the model first.")

    crop_model = joblib.load(CROP_MODEL_PATH)
    risk_model = joblib.load(RISK_MODEL_PATH)

    return crop_model, risk_model


def predict_crop_and_risk(input_data):
    """
    Predict crop recommendation and risk level from input data.
    """

    crop_model, risk_model = load_models()

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    crop_prediction = crop_model.predict(input_data)[0]
    risk_prediction = risk_model.predict(input_data)[0]

    return {
        "recommended_crop": crop_prediction,
        "risk_level": risk_prediction
    }