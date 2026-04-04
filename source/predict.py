import os
import joblib
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


def predict_crop_and_risk(input_data=None, **kwargs):
    """
    Predict crop and risk level.
    Accepts either:
    1. a DataFrame/list input_data
    OR
    2. keyword arguments like N=90, P=42, ...
    """

    # If keyword args are passed from test/API/dashboard
    if kwargs:
        input_data = pd.DataFrame([{
            "N": kwargs["N"],
            "P": kwargs["P"],
            "K": kwargs["K"],
            "temperature": kwargs["temperature"],
            "humidity": kwargs["humidity"],
            "ph": kwargs["ph"],
            "rainfall": kwargs["rainfall"]
        }])

    elif input_data is None:
        raise ValueError("No input data provided for prediction.")

    crop_model, risk_model = load_models()

    crop_prediction = crop_model.predict(input_data)[0]
    risk_prediction = risk_model.predict(input_data)[0]

    advisory = generate_advisory(
        ph=input_data.iloc[0]["ph"],
        rainfall=input_data.iloc[0]["rainfall"]
    )

    return {
        "recommended_crop": crop_prediction,
        "risk_level": risk_prediction,
        "advisory": advisory
    }