from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

app = FastAPI(title="Turbofan RUL Prediction API", version="1.0.0")

# Load trained model
MODEL_PATH = 'models/saved_models/lightgbm_rul_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except:
    print("Warning: Model not found. Please train a model first.")
    model = None

class SensorData(BaseModel):
    unit_id: int
    time_cycle: int
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

class PredictionResponse(BaseModel):
    unit_id: int
    predicted_rul: float
    maintenance_required: bool
    confidence_level: str

@app.get("/")
def root():
    return {"message": "Turbofan RUL Prediction API", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
def predict_rul(data: SensorData):
    """
    Predict Remaining Useful Life for a turbofan engine.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    try:
        prediction = model.predict(input_df.drop(['unit_id', 'time_cycle'], axis=1))[0]
        
        # Determine maintenance requirement (threshold: 30 cycles)
        maintenance_required = prediction < 30
        
        # Confidence level based on prediction value
        if prediction < 15:
            confidence = "high"
        elif prediction < 50:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            unit_id=data.unit_id,
            predicted_rul=float(prediction),
            maintenance_required=maintenance_required,
            confidence_level=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
