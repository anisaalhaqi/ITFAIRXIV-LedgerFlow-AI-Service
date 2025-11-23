import os
import json
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from app.analysis_logic import run_analysis, Transaction

app = FastAPI(
    title="LedgerFlow AI Service (Gemini)",
    description="API for fetching AI-powered financial forecasts and scores using Gemini LLM.",
    version="1.0.0"
)

@app.post("/calculate-score", response_model=dict)
async def calculate_score(transactions: list[Transaction], current_balance: float = 0.0):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. AI service is unavailable."
        )

    try:
        result = run_analysis(transactions, current_balance)
        return result
        
    except Exception as e:
        print(f"Analysis or API call failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal AI Analysis Failed during computation or Gemini API call."
        )

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LedgerFlow AI"}