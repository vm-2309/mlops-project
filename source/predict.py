import joblib
import pandas as pd
from source.data_processing import create_advisory


# Load saved models
crop_model = joblib.load("models/crop_model.pkl")
risk_model = joblib.load("models/risk_model.pkl")


def predict_crop_and_risk(N, P, K, temperature, humidity, ph, rainfall):
    input_data = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    input_df = pd.DataFrame([input_data])

    crop_prediction = crop_model.predict(input_df)[0]
    risk_prediction = risk_model.predict(input_df)[0]
    advisory = create_advisory(input_data)

    result = {
        "recommended_crop": crop_prediction,
        "risk_level": risk_prediction,
        "advisory": advisory
    }

    return result


if __name__ == "__main__":
    result = predict_crop_and_risk(
        N=90,
        P=42,
        K=43,
        temperature=20.8,
        humidity=82.0,
        ph=6.5,
        rainfall=202.9
    )

    print("Recommended Crop:", result["recommended_crop"])
    print("Risk Level:", result["risk_level"])
    print("Advisory:", result["advisory"])