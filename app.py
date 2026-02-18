from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Project")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

class InputFeature(BaseModel):
   
    income: int
    credit_score: int
    loan_amount: int
    years_employed: int
    points: float

@app.get("/")
def home():
    return {"message": "API is running successfully 🚀"}

@app.post("/predict")
def predict(data: InputFeature):
    try:
        features = np.array([
            
            data.income,
            data.credit_score,
            data.loan_amount,
            data.years_employed,
            data.points
        ]).reshape(1, -1)

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
