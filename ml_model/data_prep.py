from schemas.schemas import PredictionInput
from config.settings import DATA_TRAIN_FILE_PATH, PRODUCT_NAME, STORE_NUMBER
import pandas as pd


def preprocess_input(input_data: PredictionInput):
    ds = pd.read_csv(DATA_TRAIN_FILE_PATH)
    ds = ds[(ds["family"] == PRODUCT_NAME) & (ds["store_nbr"] == STORE_NUMBER)]
    ds = ds[["date", "sales"]]
    ds["date"] = pd.to_datetime(ds["date"], format="%Y-%m-%d")
    ds = ds.sort_values("date")
    ds = ds.rename(columns={"date": "ds", "sales": "y"})
    print(ds)
    return ds
