# utils/speech_utils.py

import os
import tempfile
from google.cloud import texttospeech
import openai
import streamlit as st

# ---------------------------------------------------------
# Google TTS Client (cached via Streamlit cache)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_tts_client():
    return texttospeech.TextToSpeechClient()

def synthesize_text_to_mp3(text: str, voice_name: str = "en-US-Wavenet-D", speaking_rate: float = 1.0) -> bytes:
    """
    Convert input text to an MP3 byte string using Google TTS.
    Returns raw MP3 bytes.
    """
    client = get_tts_client()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speaking_rate)
    try:
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Google TTS Error: {e}")
        return b""

# ---------------------------------------------------------
# OpenAI Whisper Transcription
# ---------------------------------------------------------
def transcribe_whisper(audio_bytes: bytes, model_name: str = "whisper-1") -> str:
    """
    Transcribe given audio bytes (WAV/MP3) via OpenAI Whisper API.
    Returns the transcript text.
    """
    try:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name

        audio_file = open(tmp_path, "rb")
        transcript = openai.Audio.transcribe(model=model_name, file=audio_file)
        audio_file.close()
        os.remove(tmp_path)
        return transcript.get("text", "")
    except Exception as e:
        st.error(f"Whisper ASR Error: {e}")
        return ""
