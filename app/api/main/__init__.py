from fastapi import APIRouter
from app.api.main.controller import llm_controller

api_router = APIRouter()
api_router.include_router(llm_controller.router)
