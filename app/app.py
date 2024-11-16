from fastapi import FastAPI
from app.core.config import settings
from app.api.main import api_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

    app.include_router(api_router, prefix=settings.API_PREFIX")

    return app

app = create_app()

@app.get("/")
def read_root():
    return {"message": "LLM service is running!"}
