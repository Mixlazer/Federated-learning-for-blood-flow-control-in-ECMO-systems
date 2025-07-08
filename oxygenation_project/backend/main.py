from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import datetime
import uuid
import random
import mlflow.sklearn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Federated Oxygenation System API is running. Visit /docs for Swagger UI."}

# Симуляция хранилища версий
model_versions = [
    {
        "id": "1.0",
        "created_at": "2025-07-03",
        "accuracy": 0.92,
    },
    {
        "id": "1.1",
        "created_at": "2025-07-10",
        "accuracy": 0.94,
    }
]

class PredictionRequest(BaseModel):
    age: int
    gender: str  # "male" or "female"
    height: float
    weight: float
    pump_type: str

class PredictionResponse(BaseModel):
    predicted_rpm: float

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    multiplier = 1.0 if data.gender == "male" else 0.9
    type_modifier = {
        "type1": 1.0,
        "type2": 1.2,
        "type3": 0.8
    }.get(data.pump_type, 1.0)

    rpm = (data.weight * data.height / max(data.age, 1)) * multiplier * type_modifier
    return {"predicted_rpm": round(rpm, 2)}

@app.post("/upload-data")
def upload_training_data(file: UploadFile = File(...)):
    new_version_id = f"1.{len(model_versions)}"
    new_accuracy = round(0.92 + random.uniform(0.01, 0.05), 2)
    model_versions.append({
        "id": new_version_id,
        "created_at": datetime.date.today().isoformat(),
        "accuracy": new_accuracy
    })
    return {"message": "Data uploaded successfully", "new_version": new_version_id}

@app.get("/versions")
def get_versions():
    return model_versions
