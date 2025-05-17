# app/services/language_detection.py
"""
Language detection for Luna AI transcription service
"""
import logging
import asyncio
import subprocess
from typing import Optional, Dict, Any, List
import os
import tempfile
import json

logger = logging.getLogger("transcription.language")

class LanguageDetector:
    """
    Detects language in audio files for improved transcription accuracy
    """
    def __init__(self):
        self.language_codes = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "ur": "Urdu",
            "te": "Telugu",
            "ta": "Tamil",
            "mr": "Marathi",
            "gu": "Gujarati"
            # Add more languages as needed
        }
    
    async def detect_from_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect language from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detected language information
        """
        logger.info(f"Detecting language in audio file: {audio_path}")
        
        # Try different methods in order
        methods = [
            self.detect_with_assemblyai,
            self.detect_with_whispercpp,
            self.detect_with_vosk,
            self.detect_with_pydub_analysis
        ]
        
        for method in methods:
            try:
                result = await method(audio_path)
                if result and result.get("language_code"):
                    return result
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {str(e)}")
        
        # If all methods fail, return default
        return {
            "language_code": "en",
            "language_name": "English",
            "confidence": 0.5,
            "method": "default"
        }
    
    async def detect_with_assemblyai(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect language using AssemblyAI
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detected language
        """
        try:
            # Check if AssemblyAI API key is configured
            api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not api_key:
                return None
                
            try:
                import assemblyai as aai
            except ImportError:
                logger.error("AssemblyAI library not installed. Run: pip install assemblyai")
                return None
            
            # Set API key
            aai.settings.api_key = api_key
            
            # Run in a thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            
            def process():
                # Create transcriber
                transcriber = aai.Transcriber()
                
                # Configure for language detection with minimal processing
                config = {
                    "language_detection": True,
                    "punctuate": False,
                    "format_text": False,
                    "disfluencies": False,
                    "auto_highlights": False,
                    "auto_chapters": False
                }
                
                # Upload audio
                transcript = transcriber.transcribe(audio_path, **config)
                
                if hasattr(transcript, 'language_code') and transcript.language_code:
                    return {
                        "language_code": transcript.language_code,
                        "language_name": self.language_codes.get(
                            transcript.language_code, 
                            transcript.language_code
                        ),
                        "confidence": 0.9,  # AssemblyAI is pretty reliable
                        "method": "assemblyai"
                    }
                return None
            
            result = await loop.run_in_executor(None, process)
            return result
            
        except Exception as e:
            logger.error(f"Error detecting language with AssemblyAI: {str(e)}")
            return None
    
    async def detect_with_whispercpp(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect language using whisper.cpp (if installed)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detected language
        """
        try:
            # Check if whisper.cpp is installed (just the whisper command)
            try:
                subprocess.run(["whisper", "--help"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
            except Exception:
                logger.warning("whisper.cpp not found on the system")
                return None
            
            # Use whisper.cpp to detect language (it outputs language detection first)
            # Just process the first 30 seconds for speed
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", "0",
                "-t", "30",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                "-"
            ]
            
            ffmpeg_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            whisper_cmd = [
                "whisper",
                "--language", "auto",
                "--model", "tiny",
                "-"
            ]
            
            whisper_process = subprocess.Popen(
                whisper_cmd,
                stdin=ffmpeg_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Close ffmpeg's stdout to allow whisper to receive EOF
            ffmpeg_process.stdout.close()
            
            # Get output
            stdout, stderr = whisper_process.communicate()
            output = stdout.decode("utf-8")
            
            # Parse output for language
            for line in output.split("\n"):
                if "Detected language:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        language_info = parts[1].strip()
                        language_code = language_info.split(" ")[0].strip()
                        
                        return {
                            "language_code": language_code,
                            "language_name": self.language_codes.get(language_code, language_code),
                            "confidence": 0.8,
                            "method": "whisper.cpp"
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting language with whisper.cpp: {str(e)}")
            return None
    
    async def detect_with_vosk(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect language using Vosk (if installed)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detected language
        """
        try:
            # Check if Vosk is installed
            try:
                import vosk
            except ImportError:
                logger.warning("Vosk not installed. Run: pip install vosk")
                return None
            
            # Use Vosk for language identification
            from vosk import Model, KaldiRecognizer, SetLogLevel
            import wave
            
            # Quiet mode
            SetLogLevel(-1)
            
            # Try to find a vosk-model-small-XX model directory
            model_path = None
            model_dirs = ["vosk-model-small-en", "vosk-model-small-cn", "vosk-model-small-fr"]
            
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    model_path = model_dir
                    break
            
            if not model_path:
                return None
            
            # Convert to 16kHz mono WAV if needed
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_path = temp_wav.name
            
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                temp_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Process with Vosk
            wf = wave.open(temp_path, "rb")
            
            # Load model
            model = Model(model_path)
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(False)
            
            # Read in chunks
            chunk_size = 4000
            data = wf.readframes(chunk_size)
            
            while len(data) > 0:
                rec.AcceptWaveform(data)
                data = wf.readframes(chunk_size)
            
            # Get final result
            result = json.loads(rec.FinalResult())
            
            # Clean up
            os.unlink(temp_path)
            
            # Extract language indication from result
            if "text" in result and result["text"]:
                # Vosk doesn't give direct language info, but if text is recognized with an English model,
                # it's likely English
                return {
                    "language_code": "en",  # Based on model used
                    "language_name": "English",
                    "confidence": 0.7,
                    "method": "vosk"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting language with Vosk: {str(e)}")
            return None
    
    async def detect_with_pydub_analysis(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Basic audio analysis with pydub for language detection clues
        This is very basic and just looks at audio patterns
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detected language (very low confidence)
        """
        try:
            # Check if pydub is installed
            try:
                from pydub import AudioSegment
                import numpy as np
            except ImportError:
                logger.warning("pydub not installed. Run: pip install pydub numpy")
                return None
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Extract features
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate
            
            # Calculate simple signal statistics
            mean = np.mean(samples)
            std = np.std(samples)
            
            # Calculate zero crossing rate (ZCR)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(samples)))) / len(samples)
            
            # Very basic heuristics:
            # - Languages like Japanese and Korean have higher ZCR
            # - Tonal languages like Chinese have specific patterns
            # - Just using some basic thresholds here
            
            language_code = "en"  # Default to English
            confidence = 0.3  # Very low confidence
            
            if zero_crossings > 0.1:  # Higher ZCR
                language_code = "ja"  # Japanese as a guess
            elif std > 10000:  # Higher variability
                language_code = "zh"  # Chinese as a guess
            
            return {
                "language_code": language_code,
                "language_name": self.language_codes.get(language_code, language_code),
                "confidence": confidence,
                "method": "audio_analysis"
            }
            
        except Exception as e:
            logger.error(f"Error detecting language with audio analysis: {str(e)}")
            return None