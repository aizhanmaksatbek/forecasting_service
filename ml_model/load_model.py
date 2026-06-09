import torch


async def load_model():
    return torch.load(
            "checkpoint/gnn_tft_best.pt",
            map_location=torch.device("cpu")
        )
