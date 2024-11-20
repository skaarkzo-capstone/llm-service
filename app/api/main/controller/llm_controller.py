from fastapi import APIRouter, Depends
from app.model.request.prompt_request import PromptRequest
from app.service.llm_service import generate
from app.service.llm_service import chat

router = APIRouter()

@router.post("/generate")
async def generate_prompt(prompt: PromptRequest):
    return generate(prompt)

@router.post("/chat")
async def generate_chat(content: PromptRequest):
    return chat(content)
