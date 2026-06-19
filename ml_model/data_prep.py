from schemas.schemas import PurchaseRecord
import pandas as pd
from config.settings import DATA_FILE_PATH
import asyncio
from fastapi import BackgroundTasks


async def get_input_data(
        background_tasks: BackgroundTasks
):
    """
    This function loads the prediction dataset
    """
    async def validate_features(df: pd.DataFrame):
        await asyncio.gather(*[
            asyncio.to_thread(PurchaseRecord.model_validate, row.to_dict())
            for _, row in df.iterrows()
        ])

    df = pd.read_csv(DATA_FILE_PATH)
    background_tasks.add_task(validate_features, df)
    return df
