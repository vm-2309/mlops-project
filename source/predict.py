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


def generate_advisory(risk_level):
    """
    Generate simple advisory message based on predicted risk.
    """
    if str(risk_level).lower() == "safe":
        return "Conditions look suitable for cultivation. Proceed with recommended crop planning."
    elif str(risk_level).lower() == "moderate":
        return "Moderate agricultural risk detected. Monitor soil and weather conditions carefully."
    elif str(risk_level).lower() == "risky":
        return "High agricultural risk detected. Take preventive measures before cultivation."
    else:
        return "Risk advisory unavailable. Please review input conditions."


def predict_crop_and_risk(input_data=None, **kwargs):
    """
    Predict crop recommendation and risk level.

    Supports:
    1. Dictionary input
    2. DataFrame input
    3. Direct keyword arguments like N=90, P=42, ...
    """

    crop_model, risk_model = load_models()

    # If user passes keyword arguments like N=90, P=42...
    if kwargs:
        input_data = pd.DataFrame([kwargs])

    # If user passes a dictionary
    elif isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # If no valid input is provided
    elif input_data is None:
        raise ValueError("No input data provided for prediction.")

    crop_prediction = crop_model.predict(input_data)[0]
    risk_prediction = risk_model.predict(input_data)[0]

    advisory = generate_advisory(risk_prediction)

    return {
        "recommended_crop": crop_prediction,
        "risk_level": risk_prediction,
        "advisory": advisory
    }