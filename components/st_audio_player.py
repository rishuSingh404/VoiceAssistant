# components/st_audio_player.py

import streamlit as st
import tempfile

def play_mp3(mp3_bytes: bytes):
    """
    Given raw MP3 bytes, write them to a temp file and play via st.audio.
    """
    if not mp3_bytes:
        return
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3_bytes)
        tmp.flush()
        tmp_path = tmp.name
    st.audio(tmp_path, format="audio/mp3")
    # Optionally remove after playback
    # os.remove(tmp_path)
