from schemas.schemas import PurchaseRecord, PurchaseHistory
import pandas as pd
from config.settings import DATA_FILE_PATH


async def get_input_data():
    """
    This function loads the prediction dataset
    """
    purchase_history = PurchaseHistory()
    df = pd.read_csv(DATA_FILE_PATH)
    for _, row in df.iterrows():
        purchase_history.register_record(PurchaseRecord(row))
    return PurchaseHistory(purchase_history)
