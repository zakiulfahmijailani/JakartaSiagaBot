FROM python:3.11-slim

# System deps untuk stack geospasial
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc gdal-bin libgdal-dev libspatialindex-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cache layer)
COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Salin kode (pastikan bot.py ada di repo ini)
COPY . .

# Variabel default (boleh di-override via ENV saat runtime di SumoPod)
ENV PYTHONUNBUFFERED=1 \
    OPENROUTER_MODEL=openai/gpt-4o-mini \
    PORT=8080

# Long-polling; tidak butuh port terbuka
CMD ["python", "bot.py"]
