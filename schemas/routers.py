from schemas.schemas import PredictionInput, PredictionOutput
from fastapi import APIRouter, Request, status
from ml_model.data_prep import preprocess_input
from ml_model.predictor import Predictor

router = APIRouter()


@router.get("/health")
def health_check(request: Request):
    if request.app.state.prediction_model is not None:
        return {
            "status": status.HTTP_200_OK,
            "message": "Model is loaded and ready to use."
            }
    else:
        return {
            "status": status.HTTP_404_NOT_FOUND,
            "message": "Model is not loaded."
            }


@router.post("/model", response_model=PredictionOutput)
async def submit_prediction(
    request: Request,
    input_data: PredictionInput,
):
    input_data = preprocess_input(input_data)
    prediction = Predictor().predict(
        input_data, request.app.state.prediction_model
        )
    return prediction
