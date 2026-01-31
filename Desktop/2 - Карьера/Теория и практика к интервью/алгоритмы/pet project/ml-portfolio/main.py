from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_survival
import joblib

app = FastAPI(title="Titanic ML API")

class Passenger(BaseModel):
    pclass: int = 3
    sex: int = 0  # 0=male, 1=female
    age: float = 25.0
    sibsp: int = 0
    parch: int = 0
    fare: float = 7.8

@app.post("/predict")
def predict(passenger: Passenger):
    prob = predict_survival([[
        passenger.pclass, passenger.sex, passenger.age,
        passenger.sibsp, passenger.parch, passenger.fare
    ]])
    return {"survival_probability": prob[0], "will_survive": prob[0] > 0.5}

@app.get("/")
def root():
    return {"message": "Titanic ML API готов! POST /predict"}
