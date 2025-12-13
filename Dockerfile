# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .
# The installation process will now download torch and transformers
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your local files into the container
COPY . .

EXPOSE 7860

# Command to run your FastAPI application
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "7860"]