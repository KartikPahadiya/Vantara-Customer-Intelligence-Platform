# Vantara platform image — API + dashboard in one container for local eval.
# docker-compose.yml orchestrates this with PostgreSQL.
FROM python:3.12-slim

WORKDIR /app

# System deps for LightGBM/psycopg2 wheels are prebuilt; keep this minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch from the PyTorch index (requirements pins +cpu builds).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate docs diagrams at build time (idempotent, no data needed).
RUN python scripts/make_diagrams.py && python scripts/make_notebooks.py || true

EXPOSE 8000 8501

# Default: run the API. The dashboard is a separate compose service
# overriding the command.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
