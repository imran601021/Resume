FROM python:3.9-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (much faster resolver/installer than pip)
RUN pip install --no-cache-dir uv

COPY requirements.txt .

# CPU-only torch first — avoids pulling ~3-5GB of unused CUDA/NVIDIA libraries
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies — uv will see torch is already satisfied and skip re-resolving the GPU version
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download model during build so first startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

COPY app.py agent.py .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

