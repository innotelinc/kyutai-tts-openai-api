#!/usr/bin/env python3
"""
Quick test script for OpenAI-compatible Kyutai TTS API
Tests the proper Python API integration (following Colab notebook approach)
"""

from openai import OpenAI
import time
import requests

def test_api():
    print("🧪 Testing OpenAI-compatible Kyutai TTS API")
    print("Using proper Kyutai TTS Python API (not CLI)")
    print("=" * 50)
    
    # Wait for server to be ready
    print("Waiting for server to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✓ Server is ready!")
                break
        except:
            pass
        time.sleep(1)
        print(f"Waiting... ({i+1}/30)")
    else:
        print("❌ Server not responding after 30 seconds")
        return
    
    # Initialize client
    client = OpenAI(
        api_key="dummy-key",
        base_url="http://localhost:8000/v1"
    )
    
    try:
        print("\n1. Testing health endpoint...")
        health = requests.get("http://localhost:8000/health").json()
        print(f"✓ Health: {health}")
        
        print("\n2. Testing models endpoint...")
        models = requests.get("http://localhost:8000/v1/models").json()
        print(f"✓ Available models: {len(models['data'])}")
        for model in models['data']:
            print(f"  - {model['id']}")
        
        print("\n3. Testing TTS generation (this may take a while for first run)...")
        print("Note: First generation will download the Kyutai TTS model (~1-2GB)")
        
        start_time = time.time()
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Hello! This is a test of the Kyutai TTS system using the proper Python API.",
            response_format="wav"
        )
        
        generation_time = time.time() - start_time
        
        # Save audio
        with open("kyutai_test.wav", "wb") as f:
            f.write(response.content)
        
        print(f"✓ TTS generation completed in {generation_time:.2f}s")
        print(f"✓ Audio saved: kyutai_test.wav ({len(response.content)} bytes)")
        
        print("\n4. Testing different voice and format...")
        response2 = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input="Testing different voice with MP3 format.",
            response_format="mp3"
        )
        
        with open("kyutai_nova.mp3", "wb") as f:
            f.write(response2.content)
        
        print(f"✓ Different format test: kyutai_nova.mp3 ({len(response2.content)} bytes)")
        
        print("\n5. Testing speed control...")
        response3 = client.audio.speech.create(
            model="tts-1",
            voice="echo",
            input="This text is being spoken at a different speed.",
            speed=1.5,
            response_format="wav"
        )
        
        with open("kyutai_speed.wav", "wb") as f:
            f.write(response3.content)
        
        print(f"✓ Speed control test: kyutai_speed.wav ({len(response3.content)} bytes)")
        
        print("\n🎉 All tests passed!")
        print("\nGenerated files:")
        print("- kyutai_test.wav (basic test)")
        print("- kyutai_nova.mp3 (different voice/format)")
        print("- kyutai_speed.wav (speed control)")
        print("\nThe API is working with proper Kyutai TTS integration!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Docker container is running: docker compose up -d")
        print("2. First run downloads models (~1-2GB) - be patient")
        print("3. Check logs: docker logs kyutai-tts-gpu")
        print("4. Ensure you have enough GPU memory (8GB+ recommended)")

if __name__ == "__main__":
    test_api()
