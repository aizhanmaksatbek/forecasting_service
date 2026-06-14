from schemas.schemas import PredictionInput
import pandas as pd
from config.settings import DATA_FILE_PATH, PRODUCT_NAME, STORE_NUMBER


async def get_input_data() -> PredictionInput:
    ds = pd.read_csv(DATA_FILE_PATH)
    ds = ds[(ds["family"] == PRODUCT_NAME) & (ds["store_nbr"] == STORE_NUMBER)]
    return ds
