from pydantic import BaseModel
from schemas import PredictionInput, PredictionOutput
import torch
from config.settings import FUTURE_HORIZON


class Predictor(BaseModel):
    def predict(
            self,
            input_data: PredictionInput,
            model: torch.nn.Module) -> PredictionOutput:
        print(type(model))
        return PredictionOutput(prediction=[23.5]*FUTURE_HORIZON)
