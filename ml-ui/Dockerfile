FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=$HOME/app/hf
ENV TRANSFORMERS_CACHE=$HOME/app/hf/tmp/cache


CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]
