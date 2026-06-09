from fastapi import FastAPI
from schemas import PredictionInput, PredictionOutput


app = FastAPI()


@app.get("/model/{item_id}", response_model=dict)
def read_results(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.post("/model", response_model=PredictionOutput)
async def submit_prediction(input_data: PredictionInput):
    return PredictionOutput(prediction=42.0)
