from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from inference import load_model

app = FastAPI()
model = load_model()

class InputData(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)
    return {
        "churn_prediction": int(prediction[0]),
        "model_version": "v1"
    }