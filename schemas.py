from pydantic import BaseModel, field_validator
import torch
from config.settings import FUTURE_HORIZON, PAST_DAYS


class ProductFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float


class PredictionInput(BaseModel):
    past_demand: list[ProductFeatures]

    @field_validator("past_demand")
    def check_past_days_count(cls, past_demand):
        if len(past_demand) != PAST_DAYS:
            raise ValueError(
                f"Past demand must contain {PAST_DAYS} days of history."
                )
        return past_demand


class PredictionOutput(BaseModel):
    prediction: list[float]

    @field_validator("prediction")
    def check_future_horizon_length(cls, prediction):
        if len(prediction) != FUTURE_HORIZON:
            raise ValueError(
                f"Prediction must contain {FUTURE_HORIZON} days of forecast."
                )
        return [prediction]


class Predictor(BaseModel):
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        model = torch.load(
            "checkpoint/gnn_tft_best.pt",
            map_location=torch.device("cpu")
        )
        print(type(model))
        return PredictionOutput(prediction=[23.5]*FUTURE_HORIZON)
