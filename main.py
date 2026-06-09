from fastapi import FastAPI
from schemas import PredictionInput, PredictionOutput
import torch


app = FastAPI()


@app.get("/model/{item_id}", response_model=dict)
def read_results(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.post("/model", response_model=PredictionOutput)
async def submit_prediction(input_data: PredictionInput):
    model = torch.load(
        "checkpoint/gnn_tft_best.pt",
        map_location=torch.device("cpu")
        )
    print(type(model))
    return PredictionOutput(prediction=42.0)
