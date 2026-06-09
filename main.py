from fastapi import FastAPI
from schemas import PredictionInput, PredictionOutput, Predictor

app = FastAPI()


@app.post("/model", response_model=PredictionOutput)
async def submit_prediction(input_data: PredictionInput):
    features = [input_data.feature1, input_data.feature2, input_data.feature3]
    prediction = Predictor().predict(features)
    return prediction
