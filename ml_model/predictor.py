from pydantic import BaseModel
from schemas.schemas import PurchaseHistory, PredictionOutput
from config.settings import FUTURE_HORIZON
from ml_model.TFT.tft_eval import make_forecast


class Predictor(BaseModel):
    def predict(self, input_data: PurchaseHistory) -> PredictionOutput:
        make_forecast(input_data)
        return PredictionOutput(prediction=[23.5]*FUTURE_HORIZON)
