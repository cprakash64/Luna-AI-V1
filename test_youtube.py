#!/usr/bin/env python3
import asyncio
import os
import sys

async def test_download(url):
    print(f"Testing download of: {url}")
    output = "test_download.wav"
    
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output,
        url
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    print(f"Return code: {process.returncode}")
    if stderr:
        print(f"Error: {stderr.decode()}")
    
    if os.path.exists(output):
        print(f"Success! File size: {os.path.getsize(output)} bytes")
    else:
        print("Failed: No output file")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_youtube.py <youtube-url>")
        sys.exit(1)
    
    asyncio.run(test_download(sys.argv[1]))