from fastapi import APIRouter, Depends, Body
from app.service.llm_service import evaluate, summarize_large_text_once

router = APIRouter()

@router.post("/evaluate")
async def generate_chat(content: dict = Body(...)):
    return evaluate(content)

@router.post("/summarize")
async def summarize(prompt: dict = Body(...)):
    return summarize_large_text_once(prompt)
