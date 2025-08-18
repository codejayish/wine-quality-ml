FROM python:3.10-slim

# Set working directory inside container
WORKDIR /opt/ml

# Install requirements
COPY container-requirements.txt .
RUN pip install --no-cache-dir -r container-requirements.txt

# Copy program code
COPY program/ /opt/ml/program/

# Default command runs the inference server
CMD ["python", "/opt/ml/program/serve.py"]
