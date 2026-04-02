from fastapi import FastAPI
from pydantic import BaseModel
from source.predict import predict_crop_and_risk

app = FastAPI(title="Smart Crop Prediction API")


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/")
def home():
    return {
        "message": "Smart Crop Recommendation & Risk Advisory API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: CropInput):
    result = predict_crop_and_risk(
        N=data.N,
        P=data.P,
        K=data.K,
        temperature=data.temperature,
        humidity=data.humidity,
        ph=data.ph,
        rainfall=data.rainfall
    )
    return result