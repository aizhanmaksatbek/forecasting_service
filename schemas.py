from pydantic import BaseModel
import torch


class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float


class PredictionOutput(BaseModel):
    prediction: float


class Predictor(BaseModel):
    def predict(self, input_data: dict) -> PredictionOutput:
        model = torch.load(
            "checkpoint/gnn_tft_best.pt",
            map_location=torch.device("cpu")
        )
        print(type(model))
        return PredictionOutput(prediction=42.0)
