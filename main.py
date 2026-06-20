from fastapi import FastAPI
from schemas.routers import router
import uvicorn
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI(description="API for Demand Forecasting")
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
