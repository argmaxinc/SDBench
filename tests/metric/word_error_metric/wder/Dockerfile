FROM python:3.9-slim as builder

# Install build dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir git+https://github.com/google/speaker-id.git#subdirectory=DiarizationLM

# Start a new stage with minimal image
FROM python:3.9-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy only the necessary script
COPY wder_reference.py .

# Make the script executable
RUN chmod +x wder_reference.py

# Set the entrypoint to run the script
ENTRYPOINT ["python", "/app/wder_reference.py"]
