# Use a Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your local files into the container
COPY . .

# Expose port 7860 (Hugging Face requirement)
EXPOSE 7860

# Command to run your FastAPI application using the correct port 
# Make sure "app_fastapi:app" matches your Python file name and FastAPI instance name
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "7860"]