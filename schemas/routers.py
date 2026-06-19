from fastapi import APIRouter, Depends
from ml_model.data_prep import get_input_data
from ml_model.predictor import Predictor
from typing import Annotated
import pandas as pd
from fastapi import BackgroundTasks


router = APIRouter()


@router.post("/model")
async def submit_prediction(
    input_data: Annotated[pd.DataFrame, Depends(get_input_data)],
    background_task: BackgroundTasks
):
    background_task.add_task(Predictor().predict, input_data)
    return {"message": "Started forecasting"}
