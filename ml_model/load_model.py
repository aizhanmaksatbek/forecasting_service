import torch
from config.settings import ML_MODEL_DIR


async def load_model():
    return torch.load(
            f"{ML_MODEL_DIR}tft_best_train_final.pt",
            map_location=torch.device("cpu")
        )
