# ============================
# 1º estágio: Builder
# ============================
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /install

# Dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only compatível com transformers recentes
RUN pip install --no-cache-dir torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copia requirements e instala dependências (fixando numpy<2)
COPY requirements-dk.txt .
RUN pip install --no-cache-dir -r requirements-dk.txt

# Congela dependências para reprodutibilidade
RUN pip freeze > /install/requirements.lock

# ============================
# 2º estágio: Runtime
# ============================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

WORKDIR /app

# Copia bibliotecas do builder e o lock
COPY --from=builder /usr/local /usr/local
COPY --from=builder /install/requirements.lock /app/requirements.lock

# Copia código e dados necessários
COPY src/ /app/src/
COPY app/ /app/app/
COPY data/ /app/data/

EXPOSE 8501

# Rodar Streamlit acessível pelo host e mostrar localhost no log
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]





