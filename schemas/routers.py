from schemas.schemas import PredictionOutput
from fastapi import APIRouter, Depends
from ml_model.data_prep import get_input_data
from ml_model.predictor import Predictor
from typing import Annotated
import pandas as pd

router = APIRouter()


@router.post("/model", response_model=PredictionOutput)
async def submit_prediction(
    input_data: Annotated[pd.DataFrame, Depends(get_input_data)],
):
    prediction = Predictor().predict(input_data)
    return prediction
