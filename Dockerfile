# CarbonWise Allocator - one image for both the API and the Streamlit UI.
# docker-compose.yml starts it twice with different commands.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first, so code changes do not reinstall packages.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code: our app + UI, and the teammates' data module/optimizer (unchanged).
COPY app ./app
COPY ui ./ui
COPY carbonwise-data ./carbonwise-data

# Fail the build early if PuLP's bundled CBC solver cannot run on this CPU.
RUN python -c "import app.bootstrap; from optimization import solver_available; \
assert solver_available(), 'CBC solver unavailable in image'"

# SQLite lives on a volume mounted here (see docker-compose.yml).
RUN mkdir -p /data
ENV CARBONWISE_DB_PATH=/data/carbonwise.sqlite3

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
