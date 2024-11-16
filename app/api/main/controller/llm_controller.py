from fastapi import APIRouter, Depends
from app.model.request.prompt_request import PromptRequest
from app.service.llm_service import generate

router = APIRouter()

@router.post("/generate")
async def generate_prompt(prompt: PromptRequest):
    return generate(prompt)
