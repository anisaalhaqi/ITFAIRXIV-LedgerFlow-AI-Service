FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1
ENV GEMINI_API_KEY=""

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app
COPY main.py .

Expose port yang digunakan Uvicorn

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]