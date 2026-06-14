from schemas.schemas import PurchaseRecord, PurchaseHistory
import pandas as pd
from config.settings import DATA_FILE_PATH


async def get_input_data():
    """
    This function loads the prediction dataset
    """
    df = pd.read_csv(DATA_FILE_PATH)
    purchase_history = PurchaseHistory()

    for _, row in df.iterrows():
        record = PurchaseRecord.model_validate(row.to_dict())
        purchase_history.register_record(record)
    return purchase_history
