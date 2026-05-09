FROM rocm/dev-ubuntu-22.04:6.1-complete
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y python3.11 python3-pip git curl

# ROCm PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Project deps
COPY pyproject.toml .
RUN pip install -e .

COPY . .

EXPOSE 8001

CMD ["python", "-m", "contextforge.main"]