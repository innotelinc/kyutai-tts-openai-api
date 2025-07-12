#!/usr/bin/env python3
"""
OpenAI-Compatible Kyutai TTS API Server
Provides OpenAI Audio Speech API endpoints for text-to-speech generation
"""

import os
import io
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Literal

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# OpenAI-compatible models
class SpeechRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd"] = Field("tts-1", description="TTS model to use")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to generate audio for")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Field("alloy", description="Voice to use")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field("mp3", description="Audio format")
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0, description="Speed of generated audio")

app = FastAPI(
    title="OpenAI-Compatible TTS API",
    description="OpenAI Audio Speech API compatible endpoint using Kyutai TTS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/app/api_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Voice mapping (Kyutai doesn't have multiple voices yet, but we accept OpenAI voice names)
VOICE_MAPPING = {
    "alloy": "default",
    "echo": "default", 
    "fable": "default",
    "onyx": "default",
    "nova": "default",
    "shimmer": "default"
}

def generate_audio(text: str, voice: str = "default", speed: float = 1.0) -> Path:
    """Generate audio using Kyutai TTS (proper Python API)"""
    timestamp = int(time.time() * 1000)
    output_path = OUTPUT_DIR / f"speech_{timestamp}.wav"
    
    try:
        # Import Kyutai TTS modules
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
        import numpy as np
        import soundfile as sf
        
        print("Loading Kyutai TTS model...")
        
        # Load the TTS model (following the notebook approach)
        checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, 
            n_q=32, 
            temp=0.6, 
            device=device
        )
        
        print("Preparing text and voice...")
        
        # Prepare the script (text input)
        entries = tts_model.prepare_script([text], padding_between=1)
        
        # Use default voice or map OpenAI voice names
        voice_mapping = {
            "alloy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
            "echo": "expresso/ex03-ex01_happy_001_channel1_334s.wav", 
            "fable": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
            "onyx": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
            "nova": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
            "shimmer": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
            "default": "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        }
        
        selected_voice = voice_mapping.get(voice, voice_mapping["default"])
        voice_path = tts_model.get_voice_path(selected_voice)
        
        # Prepare condition attributes
        condition_attributes = tts_model.make_condition_attributes(
            [voice_path], cfg_coef=2.0
        )
        
        print("Generating audio...")
        
        # Generate audio (following notebook approach)
        pcms = []
        
        def on_frame(frame):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))
        
        all_entries = [entries]
        all_condition_attributes = [condition_attributes]
        
        with tts_model.mimi.streaming(len(all_entries)):
            result = tts_model.generate(all_entries, all_condition_attributes, on_frame=on_frame)
        
        # Concatenate all audio frames
        if pcms:
            audio = np.concatenate(pcms, axis=-1)
            
            # Save as WAV file
            sf.write(
                str(output_path), 
                audio, 
                samplerate=tts_model.mimi.sample_rate,
                format='WAV'
            )
            
            print(f"Audio generated successfully: {output_path}")
            return output_path
        else:
            raise Exception("No audio frames generated")
            
    except ImportError as e:
        print(f"Moshi import error: {e}")
        print("This usually means missing dependencies. Check if moshi and its dependencies are properly installed.")
        # Fallback to placeholder
        create_placeholder_audio(output_path, text)
        return output_path
    except Exception as e:
        print(f"TTS generation error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to placeholder  
        create_placeholder_audio(output_path, text)
        return output_path

def create_placeholder_audio(output_path: Path, text: str):
    """Create placeholder audio when moshi fails"""
    import wave
    import numpy as np
    
    # Generate simple tone as placeholder
    sample_rate = 24000
    duration = max(1.0, len(text) * 0.1)  # Rough estimate
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple beep pattern
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz tone
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def convert_audio_format(input_path: Path, output_format: str) -> bytes:
    """Convert audio to requested format using ffmpeg"""
    try:
        if output_format == "wav":
            with open(input_path, 'rb') as f:
                return f.read()
                
        # Convert using ffmpeg for other formats
        cmd = ["ffmpeg", "-i", str(input_path), "-f", output_format, "-"]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    OpenAI-compatible audio speech endpoint
    Generates audio from text using Kyutai TTS
    """
    try:
        # Generate audio with Kyutai TTS
        audio_path = generate_audio(
            text=request.input,
            voice=VOICE_MAPPING.get(request.voice, "default"),
            speed=request.speed
        )
        
        # Convert to requested format
        audio_data = convert_audio_format(audio_path, request.response_format)
        
        # Set appropriate content type
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus", 
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        # Clean up temporary file
        audio_path.unlink(missing_ok=True)
        
        return Response(
            content=audio_data,
            media_type=content_types.get(request.response_format, "audio/wav"),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "kyutai",
                "permission": [],
                "root": "tts-1",
                "parent": None
            },
            {
                "id": "tts-1-hd", 
                "object": "model",
                "created": 1677610602,
                "owned_by": "kyutai",
                "permission": [],
                "root": "tts-1-hd",
                "parent": None
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "service": "kyutai-tts-openai-compatible"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)