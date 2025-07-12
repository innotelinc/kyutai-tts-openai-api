# Kyutai TTS OpenAI-Compatible API

A Docker-based OpenAI-compatible Text-to-Speech API server powered by Kyutai's TTS models with GPU acceleration support.

## Features

- OpenAI-Compatible API endpoints
- GPU acceleration with CUDA support
- Multiple audio formats (MP3, WAV, FLAC, AAC, Opus, PCM)
- Voice selection and speed control
- Complete Docker solution

## Quick Start

### Prerequisites
- Docker with Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ GPU memory, 16GB+ system RAM

### Installation

```bash
# Clone repository
git clone https://github.com/dwain-barnes/kyutai-tts-openai-api.git
cd kyutai-tts-openai-api

mkdir input output cache scripts

# Build and start
docker compose build --no-cache
docker compose up -d

# Test the API
python api_test_script.py
```

## Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:8000/v1"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello! This is Kyutai TTS speaking.",
    response_format="wav"
)

with open("speech.wav", "wb") as f:
    f.write(response.content)
```

### cURL

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello world!",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

## API Reference

### POST /v1/audio/speech

Generate audio from text input.

**Request:**
```json
{
  "model": "tts-1",
  "input": "Text to synthesize",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Parameters:**
- `model`: "tts-1" or "tts-1-hd" (required)
- `input`: Text 1-4096 characters (required)
- `voice`: alloy, echo, fable, onyx, nova, shimmer (optional)
- `response_format`: mp3, wav, flac, aac, opus, pcm (optional)
- `speed`: 0.25-4.0 (optional)

### GET /v1/models

List available models.

### GET /health

Health check endpoint.

## Project Structure

```
kyutai-tts-openai-api/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── api_server.py
├── dependency_check.py
├── api_test_script.py
├── input/
├── output/
├── cache/
└── scripts/
```

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Model cache directory

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
docker logs kyutai-tts-gpu
```

**Model download fails:**
- Check internet connection
- Ensure sufficient disk space (2GB+ needed)

**Slow generation:**
```bash
# Verify GPU usage
docker exec kyutai-tts-gpu python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Dependencies missing:**
```bash
docker exec kyutai-tts-gpu python dependency_check.py
```

## Security Notes

- Designed for local/internal use
- No authentication implemented
- Add proper auth for production use
- No rate limiting

## Kyutai License
The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend. The web client code is provided under the MIT license. Note that parts of this code is based on AudioCraft, released under the MIT license.
The weights for the speech-to-text models are released under the CC-BY 4.0 license.


## Acknowledgments

- [Kyutai Labs](https://github.com/kyutai-labs) for TTS models
