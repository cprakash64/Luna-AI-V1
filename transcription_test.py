#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

# Minimal versions of functions for testing
async def download_yt(url, output_path):
    print(f"Downloading {url} to {output_path}")
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", output_path, url]
    process = await asyncio.create_subprocess_exec(*cmd)
    await process.communicate()
    return process.returncode == 0

async def transcribe(audio_path):
    print(f"Transcribing {audio_path}")
    # Just a simulation - in real code this would call Google Speech API
    return {"text": "This is a test transcription"}

async def main(url):
    temp_dir = Path("./temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_path = temp_dir / "test_audio.wav"
    
    # Step 1: Download
    if await download_yt(url, audio_path):
        print(f"✅ Download successful: {os.path.getsize(audio_path)} bytes")
        
        # Step 2: Transcribe
        result = await transcribe(audio_path)
        print(f"✅ Transcription: {result}")
    else:
        print("❌ Download failed")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_transcription.py <youtube-url>")
        sys.exit(1)
    
    asyncio.run(main(sys.argv[1]))