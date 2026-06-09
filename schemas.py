from pydantic import BaseModel


class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float


class PredictionOutput(BaseModel):
    prediction: float
