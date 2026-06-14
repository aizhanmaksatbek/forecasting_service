from pydantic import BaseModel
from schemas.schemas import PredictionInput, PredictionOutput
from config.settings import FUTURE_HORIZON
from ml_model.TFT.tft_eval import make_forecast


class Predictor(BaseModel):
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        make_forecast(input_data)
        return PredictionOutput(prediction=[23.5]*FUTURE_HORIZON)
