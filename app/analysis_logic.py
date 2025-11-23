import os
import json
from pydantic import BaseModel
from google import genai
from google.genai import types
import pandas as pd
from datetime import datetime

# untuk data input dari BE
class Transaction(BaseModel):
    date: str
    category: str
    amount: float
    type: str

# untuk data output ke FE
class FinancialAnalysisResult(BaseModel):
    financial_score: int
    days_to_zero: int
    monthly_spending_shifts: list
    advice: str

# client inisialization
try:
    client = genai.Client()
except Exception as e:
    print("Warning: Gemini Client failed to initialize. Check GEMINI_API_KEY in .env")
    client = None


def run_analysis(transactions: list, current_balance: float):
    # 1. preprocessing
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # count average daily and debits
    debits = df[df['type'] == 'debit']['amount'].sum()
    days_elapsed = (datetime.now() - df['date'].min()).days + 1
    avg_daily_debit = debits / days_elapsed if days_elapsed > 0 else 0
    
    # forecasts days to zero
    days_to_zero_local = int(current_balance / avg_daily_debit) if avg_daily_debit > 0 else 999
    
    # formatting to string
    tx_text = df.to_string(index=False)
    
    # 2. analisis AI
    if client:
        try:
            prompt = f"""
            Anda adalah analis keuangan AI untuk LedgerFlow. Tugas Anda adalah menganalisis riwayat transaksi berikut 
            dan status saldo saat ini:
            
            --- DATA TRANSAKSI MENTAH ---
            {tx_text}
            
            --- STATUS SAAT INI ---
            Saldo Aktif: Rp{current_balance:,.2f}
            Rata-rata Pengeluaran Harian Bulan Ini: Rp{avg_daily_debit:,.2f}
            
            1. Berikan 'financial_score' (1-100) berdasarkan Net Flow dan Konsistensi.
            2. Berikan 'days_to_zero' berdasarkan data yang sudah dihitung (Days: {days_to_zero_local}).
            3. Identifikasi dua pergeseran pengeluaran MoM paling signifikan (shifts).
            4. Tulis 'advice' personal dan spesifik untuk pengguna.
            
            Berikan hasilnya HANYA dalam format JSON yang sesuai dengan skema yang diminta.
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FinancialAnalysisResult,
                ),
            )
            
            # 3. verifikasi dan kembalikan data
            gemini_output = json.loads(response.text)
            
            return FinancialAnalysisResult(**gemini_output).model_dump()
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # fallback ke hasil lokal jika API gagal
            pass

    # 4. Fallback (Jika Gemini gagal atau kunci tidak ada)
    score_fallback = min(max(50 + int(avg_daily_debit / 100000 * 5), 10), 100)
    shifts_fallback = [
        {"category": "Transportation", "change_percent": 50, "trend": "increase"},
        {"category": "Food", "change_percent": -20, "trend": "decrease"}
    ]
    advice_fallback = "Saat ini sistem AI tidak aktif. Lanjutkan dengan analisis dasar: Fokus pada pengeluaran transportasi bulan ini."
    
    return {
        "financial_score": score_fallback,
        "days_to_zero": days_to_zero_local,
        "monthly_spending_shifts": shifts_fallback,
        "advice": advice_fallback
    }