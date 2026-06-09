from fastapi import FastAPI
from schemas import PredictionInput, PredictionOutput, Predictor

app = FastAPI()


def preprocess_input(input_data: PredictionInput):
    # Implement any necessary preprocessing steps here
    return input_data


@app.post("/model", response_model=PredictionOutput)
async def submit_prediction(input_data: PredictionInput):
    input_data = preprocess_input(input_data)
    prediction = Predictor().predict(input_data)
    return prediction
