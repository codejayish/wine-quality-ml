FROM python:3.10-slim

WORKDIR /opt/ml

COPY container-requirements.txt .
RUN pip install --no-cache-dir -r container-requirements.txt

# Copy program code
COPY program/ /opt/ml/program/

# SageMaker looks for this env to decide program to run
ENV SAGEMAKER_PROGRAM=/opt/ml/program/train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/program

# When container is run manually (local docker), default to serving
CMD ["python", "/opt/ml/program/serve.py"]
