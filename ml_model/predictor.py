from pydantic import BaseModel
from schemas.schemas import PredictionInput, PredictionOutput
import torch
from config.settings import FUTURE_HORIZON
from ml_model.TFT.tft_eval import main


class Predictor(BaseModel):
    def predict(
            self,
            input_data: PredictionInput,
            model: torch.nn.Module) -> PredictionOutput:
        main()
        print(type(model))
        return PredictionOutput(prediction=[23.5]*FUTURE_HORIZON)
