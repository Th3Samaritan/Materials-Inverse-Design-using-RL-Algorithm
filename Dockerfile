FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production TORCH_NUM_THREADS=2
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt
COPY mast ./mast
COPY api ./api
COPY checkpoints/forward_model.pt checkpoints/policy.pt ./checkpoints/
RUN useradd --create-home appuser
USER appuser
CMD ["sh", "-c", "uvicorn api.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
