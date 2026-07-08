FROM python:3.11-slim

WORKDIR /app

# Install the package in editable mode so legal_rag_v1/config.py's
# Path(__file__).parent.parent resolves to /app (where config.yaml lives),
# matching local dev (`uv pip install -e .`).
COPY pyproject.toml ./
COPY legal_rag_v1 ./legal_rag_v1
COPY scripts ./scripts
COPY config.yaml ./

RUN pip install --no-cache-dir -e .

EXPOSE 6800

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6800/health')" || exit 1

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "6800"]
