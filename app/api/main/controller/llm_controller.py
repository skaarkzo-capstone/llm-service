from fastapi import APIRouter, Depends, Body
from app.service.llm_service import evaluate, summarize, financial_breakdown, evaluate_transation, pure_play

router = APIRouter()

@router.post("/evaluate")
async def generate_chat(content: dict = Body(...)):
    return pure_play(content)

@router.post("/evaluate/transaction")
async def generate_chat(content: dict = Body(...)):
    return evaluate_transation(content)

@router.post("/summarize")
async def summarize_endpoint(prompt: dict = Body(...)):
    return summarize(prompt)

@router.post("/breakdown")
async def financial(prompt: dict = Body(...)):
    return financial_breakdown(prompt)
