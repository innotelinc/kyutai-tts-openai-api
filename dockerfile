# Use Python 3.12 base image and add CUDA support
FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    sox \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (for better caching)
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support for Python 3.12
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    librosa \
    soundfile \
    huggingface_hub \
    einops \
    transformers \
    accelerate

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic

# Install moshi package with all dependencies (following Colab notebook)
RUN pip install --no-cache-dir 'sphn<0.2'
RUN pip install --no-cache-dir "moshi==0.2.8"

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/scripts /app/api_output

# Download the Kyutai delayed-streams-modeling repository
RUN git clone https://github.com/kyutai-labs/delayed-streams-modeling.git /app/kyutai-repo

# Copy the TTS script from the repository
RUN cp /app/kyutai-repo/scripts/tts_pytorch.py /app/scripts/ || echo "TTS script not found, will create custom one"

# Create a custom TTS runner script
RUN cat > /app/scripts/tts_runner.py << 'EOF'
#!/usr/bin/env python3
"""
Kyutai TTS PyTorch Runner
Dockerized implementation for text-to-speech generation
"""
import sys
import os
import argparse
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Kyutai TTS PyTorch Runner')
    parser.add_argument('input_file', help='Input text file or "-" for stdin')
    parser.add_argument('output_file', help='Output audio file')
    parser.add_argument('--model', default='kyutai/tts-1.6b-en_fr', help='TTS model to use')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Handle stdin input
    if args.input_file == '-':
        # Read from stdin and create temporary file
        text = sys.stdin.read().strip()
        temp_file = '/tmp/temp_input.txt'
        with open(temp_file, 'w') as f:
            f.write(text)
        input_file = temp_file
    else:
        input_file = args.input_file
    
    # Check if the original TTS script exists
    tts_script = Path('/app/scripts/tts_pytorch.py')
    if tts_script.exists():
        print("Using original TTS script from Kyutai repository")
        import subprocess
        cmd = ['python', str(tts_script), input_file, args.output_file]
        subprocess.run(cmd, check=True)
    else:
        print("Using moshi package for TTS generation")
        import subprocess
        cmd = [
            'python', '-m', 'moshi.run_inference', 
            '--hf-repo', args.model,
            input_file,
            args.output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        print(f"Audio generated: {args.output_file}")

if __name__ == '__main__':
    main()
EOF

# Make the script executable
RUN chmod +x /app/scripts/tts_runner.py

# Create a simple test script
RUN cat > /app/test_tts.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Kyutai TTS
"""
import sys
import torch

def test_python_version():
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 12:
        print("✓ Python 3.12+ detected")
    else:
        print("✗ Python 3.12+ required")
        return False
    return True

def test_cuda():
    print("Testing CUDA availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available. TTS will run on CPU (slower).")

def test_moshi():
    print("\nTesting moshi package...")
    try:
        import moshi
        print(f"Moshi package imported successfully")
        print(f"Moshi version: {getattr(moshi, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"Failed to import moshi: {e}")
        return False
    return True

def test_huggingface():
    print("\nTesting Hugging Face connection...")
    try:
        from huggingface_hub import hf_hub_download
        print("Hugging Face Hub available")
        # Try to check if we can access the model (without downloading)
        from huggingface_hub import repo_exists
        if repo_exists("kyutai/tts-1.6b-en_fr"):
            print("✓ Kyutai TTS model repository is accessible")
        else:
            print("⚠ Cannot verify model repository access")
    except Exception as e:
        print(f"Issue with Hugging Face Hub: {e}")

def test_api_dependencies():
    print("\nTesting API dependencies...")
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI and Uvicorn available")
        return True
    except ImportError as e:
        print(f"✗ API dependencies missing: {e}")
        return False

if __name__ == '__main__':
    python_ok = test_python_version()
    test_cuda()
    moshi_ok = test_moshi()
    test_huggingface()
    api_ok = test_api_dependencies()
    
    if python_ok and moshi_ok and api_ok:
        print("\n✓ Environment setup appears to be working!")
        print("\nTo run TTS:")
        print("echo 'Hello, this is a test.' | python scripts/tts_runner.py - /app/output/test.wav")
        print("\nTo start API server:")
        print("python api_server.py")
        print("Then visit: http://localhost:8000/docs")
    else:
        print("\n✗ Environment setup has issues. Check the logs above.")
        sys.exit(1)
EOF

RUN chmod +x /app/test_tts.py

# Copy the API server and dependency checker
COPY api_server.py /app/
COPY dependency_check.py /app/

# Create example text file
RUN echo "Hello, this is Kyutai Text-to-Speech running in Docker with GPU support!" > /app/input/example.txt

# Set default command to start API server
CMD ["python", "api_server.py"]

# Expose any ports if needed (for future web interface)
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1