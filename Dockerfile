FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv

WORKDIR /app
# dependencies first, so that code changes do not reinstall them
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --extra app

COPY app ./app
# COMMIT_SHA is written by the deploy step; the glob keeps the build working without it
COPY COMMIT_SHA* ./

# Hugging Face Spaces run containers as user 1000
RUN useradd --create-home --uid 1000 user && chown -R user /app
USER user
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:7860/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
