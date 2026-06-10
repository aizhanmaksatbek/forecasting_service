from fastapi import FastAPI
from contextlib import asynccontextmanager
from ml_model.load_model import load_model
from schemas.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_model = await load_model()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
