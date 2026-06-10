import torch


async def load_model():
    return torch.load(
            "ml_model/checkpoint/gnn_tft_best.pt",
            map_location=torch.device("cpu")
        )
