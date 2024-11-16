from fastapi import APIRouter, Depends
from app.model.dto.model_deto import TextRequest
from app.service.service import generate

router = APIRouter()

@router.post("/generate")
async def generate_prompt(prompt: PromptRequest):
    return generate(prompt)
