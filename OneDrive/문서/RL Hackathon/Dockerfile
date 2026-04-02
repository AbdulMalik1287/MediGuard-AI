FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY patient_simulator.py .
COPY mediguard_env.py .
COPY inference.py .
COPY server.py .
COPY openenv.yaml .
COPY README.md .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
