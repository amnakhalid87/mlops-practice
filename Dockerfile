# ---- Base image ----
FROM python:3.11-slim

# Working directory container ke andar
WORKDIR /app

# System dependencies (agar zaroorat pare, minimal rakha hai)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pehle sirf requirements copy karo -> Docker layer caching se builds fast hongi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki sara code copy karo
COPY . .

# Model train karo image build ke waqt (taake container ready-to-serve ho)
RUN python train.py

# Render container port (render.yaml mein PORT=8000 set hai)
EXPOSE 8000

# App start command - $PORT env var use karta hai (Render isi tarah port assign karta hai)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
