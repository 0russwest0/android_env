FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better caching
COPY pool_manager/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy manager code
COPY pool_manager /app/pool_manager

EXPOSE 8080

CMD ["python", "-m", "pool_manager.manager"]


