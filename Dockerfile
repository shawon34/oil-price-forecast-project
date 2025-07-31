FROM python:3.11-slim-bullseye

RUN apt-get update && \
    apt-get install -y build-essential libssl-dev libffi-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]