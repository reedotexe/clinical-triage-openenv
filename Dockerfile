# Clinical Triage OpenEnv Environment
# Dockerfile must be in the project ROOT (not server/) per hackathon requirements.
#
# Build:  docker build -t clinical_triage .
# Run:    docker run -p 8000:8000 clinical_triage
# HF:     openenv push --repo-id <username>/clinical-triage

FROM python:3.11-slim

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies (cached layer — copy requirements first)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Environment server listens on port 8000
# (app_port: 8000 in README matches — HF Spaces will proxy to this port)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# server.app:app resolves relative to WORKDIR (/home/user/app)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
