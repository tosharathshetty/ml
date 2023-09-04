FROM python:3.10-slim



RUN apt-get update && apt-get install -y curl


WORKDIR /app

COPY requirements-fastapi.txt ./

RUN pip install --no-cache-dir -r requirements-fastapi.txt

COPY . .


ENV HF_HOME=/app/hf
ENV TRANSFORMERS_CACHE=/app/hf/tmp/cache


CMD ["uvicorn", "endpoints:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]