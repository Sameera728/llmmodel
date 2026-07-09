version: "3.8"

services:
  # FastAPI backend service
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: open_gpt_llm_backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://admin:super_secure_pass@postgres:5432/llm_db
      - TENSORBOARD_LOG_DIR=/app/checkpoints/runs
    volumes:
      - ../checkpoints:/app/checkpoints
      - ../datasets:/app/datasets
      - ../configs:/app/configs
    depends_on:
      postgres:
        condition: service_healthy
    # Deploy GPU capabilities inside the docker runtime if NVIDIA drivers are configured
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  # Relational database for production users, prompts, and metrics
  postgres:
    image: postgres:16-alpine
    container_name: open_gpt_llm_postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=super_secure_pass
      - POSTGRES_DB=llm_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d llm_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # TensorBoard service for visual progress monitoring during fine-tuning
  tensorboard:
    image: tensorflow/tensorflow:latest-gpu
    container_name: open_gpt_llm_tensorboard
    ports:
      - "6006:6006"
    volumes:
      - ../checkpoints/runs:/logs
    command: tensorboard --logdir=/logs --host=0.0.0.0
    restart: unless-stopped

volumes:
  postgres_data:
