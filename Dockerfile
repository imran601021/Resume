FROM python:3.9-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY app.py analyzer.py agent.py .
COPY templates/ templates/
COPY static/ static/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/health
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "--timeout", "120", "--workers", "1", "app:app"]
