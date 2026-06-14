from pydantic import BaseModel, field_validator
from config.settings import FUTURE_HORIZON


class PurchaseRecord(BaseModel):
    store_nbr: int
    family: str
    date: str
    dow: int
    month: int
    weekofyear: int
    id: int
    sales: float
    onpromotion: float
    state: str
    store_type: str
    cluster: int
    transactions: float
    dcoilwtico: float
    is_holiday: int
    is_workday: int


class PurchaseHistory(BaseModel):
    past_demand: list[PurchaseRecord]

    def __init__(self):
        self.past_demand = list()

    def register_record(self, record: PurchaseRecord):
        self.past_demand.append(PurchaseRecord(record))


class PredictionOutput(BaseModel):
    prediction: list[float]

    @field_validator("prediction")
    def check_future_horizon_length(cls, prediction):
        if len(prediction) != FUTURE_HORIZON:
            raise ValueError(
                f"Prediction must contain {FUTURE_HORIZON} days of forecast."
                )
        return [prediction]
