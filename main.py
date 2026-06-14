from fastapi import FastAPI
from schemas.routers import router


app = FastAPI()
app.include_router(router)
