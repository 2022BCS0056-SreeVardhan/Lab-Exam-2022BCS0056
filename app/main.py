from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

NAME = "Sree Vardhan Reddy"
ROLL_NO = "2022BCS0056"

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Wine Quality API")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def root():
    return {
        "name": NAME,
        "roll_no": ROLL_NO,
        "message": "Wine Quality API is running"
    }

@app.post("/predict")
def predict(data: WineInput):
    features = [[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]]

    pred = float(model.predict(features)[0])
    pred = max(3, min(8, round(pred)))

    return {
        "name": NAME,
        "roll_no": ROLL_NO,
        "wine_quality": int(pred)
    }