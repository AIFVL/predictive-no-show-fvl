FROM python:3.13-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY backend/requirements.txt ./requirements-backend.txt
COPY requirements.txt ./requirements-root.txt

# Instalar dependencias Python
RUN python -m pip install --no-cache-dir -r requirements-backend.txt && \
    python -m pip install --no-cache-dir -r requirements-root.txt

# Copiar código
COPY backend ./backend
COPY data ./data
COPY outputs ./outputs
COPY src ./src
COPY configs ./configs

# Exponer puerto
EXPOSE 8000

# Comando por defecto
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
