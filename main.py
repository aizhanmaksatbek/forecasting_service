from fastapi import FastAPI, status
from schemas import PredictionInput, PredictionOutput
from contextlib import asynccontextmanager
from ml_model.pre_process import preprocess_input
from ml_model.load_model import load_model
from ml_model.predictor import Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_model = await load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/model", response_model=PredictionOutput)
async def submit_prediction(input_data: PredictionInput):
    input_data = preprocess_input(input_data)
    prediction = Predictor().predict(input_data, app.state.prediction_model)
    return prediction


@app.get("/health")
def health_check():
    if app.state.prediction_model is not None:
        return {"status": status.HTTP_200_OK, "message": "Model is loaded and ready to use."}
    else:
        return {"status": status.HTTP_404_NOT_FOUND, "message": "Model is not loaded."}
