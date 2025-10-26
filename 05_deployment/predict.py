
from fastapi import FastAPI
from typing import Dict, Any
import uvicorn
import pickle

app = FastAPI(title="MLZoomcamp HW 05")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(record):
    pred = pipeline.predict_proba(record)[0, 1]
    return float(pred)


@app.post("/predict")
def predict(record: Dict[str, Any]):
    pred = predict_single(record)
    return {
        "pred_probability": pred,
        "pred_decision": (pred >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)