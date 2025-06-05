# components/st_audio_recorder.py

import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, VideoHTMLAttributes

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class AudioRecorderProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame.to_ndarray())
        return frame

def audio_recorder(label="Click to record audio", key="audio_rec"):
    """
    Uses streamlit-webrtc to record audio in-browser.
    Returns raw WAV bytes (16kHz mono) once the user presses 'Stop'.
    """
    st.markdown(label)
    recorder_ctx = webrtc_streamer(
        key=key,
        mode="SENDONLY",
        audio_processor_factory=AudioRecorderProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
    )
    if recorder_ctx.state.playing:
        if st.button("Stop Recording", key=f"stop_{key}"):
            recorder_ctx.stop()
            audio_frames = recorder_ctx.audio_processor.audio_frames
            # Convert frames to WAV bytes
            out, sr = av.audio_frame_to_ndarray(
                audio_frames, format="s16", layout="mono"
            )
            # Write to a tempfile as WAV
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, out.flatten(), 16000, format="WAV")
                tmp_path = tmp.name
                tmp.read()  # load bytes if needed
            audio_bytes = open(tmp_path, "rb").read()
            os.remove(tmp_path)
            return audio_bytes, 16000
    return None, None
