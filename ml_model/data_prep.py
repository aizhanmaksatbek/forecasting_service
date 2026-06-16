from schemas.schemas import PurchaseRecord
import pandas as pd
from config.settings import DATA_FILE_PATH


async def get_input_data():
    """
    This function loads the prediction dataset
    """
    df = pd.read_csv(DATA_FILE_PATH)

    for _, row in df.iterrows():
        PurchaseRecord.model_validate(row.to_dict())
    return df
