from fastapi import APIRouter, Depends, Body
from app.service.llm_service import generate
from app.service.llm_service import chat, summarize

router = APIRouter()

@router.post("/generate")
async def generate_prompt(prompt):
    return generate(prompt)

@router.post("/chat")
async def generate_chat(content: dict = Body(...)):
    return chat(content)

@router.post("/summarize")
async def generate_chat(prompt: dict = Body(...)):
    return summarize(prompt)
