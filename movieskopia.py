import streamlit as st
import tempfile
import subprocess
import os
from pathlib import Path
from datetime import timedelta
import openai
from openai import OpenAI
import pysrt
import ffmpeg

# ======================================================
# KONFIGURACJA API
# ======================================================
# zakładam że w pliku .env trzymasz OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# ======================================================
# FUNKCJE POMOCNICZE
# ======================================================
def extract_audio(video_path, output_path="audio.wav"):
    """Ekstrakcja audio z wideo do formatu wav"""
    (
        ffmpeg
        .input(video_path)
        .output(output_path, ac=1, ar="16k") # mono 16kHz
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def transcribe_audio(audio_path):
    """Użycie modelu OpenAI do transkrypcji"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # SOTA model do speech-to-text
            file=audio_file,
            response_format="srt"  # od razu w formacie napisów
        )
    return transcript

def save_srt(srt_text, output_srt):
    """Zapis transkrypcji do pliku SRT"""
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt_text)

# ======================================================
# STRONA STREAMLIT
# ======================================================
st.set_page_config(page_title="Generator Napisów", layout="centered")
st.title("🎬 Generator Napisów z Wideo")

uploaded_file = st.file_uploader("Wrzuć swoje wideo (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # zapisz plik wejściowy
        input_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(uploaded_file, format="video/mp4")

        st.info("⏳ Przetwarzanie... chwilka cierpliwości.")
        
        # 1. Ekstrakcja dźwięku
        audio_path = os.path.join(temp_dir, "audio.wav")
        extract_audio(input_video_path, audio_path)

        # 2. Transkrypcja (napisy w SRT)
        transcript_srt = transcribe_audio(audio_path)

        # 3. Zapis do pliku .srt
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        save_srt(transcript_srt, srt_path)

        st.success("✅ Napisy wygenerowane!")

        # Pobranie pliku napisów
        with open(srt_path, "rb") as srt_file:
            st.download_button(
                label="📥 Pobierz napisy (SRT)",
                data=srt_file,
                file_name="napisy.srt",
                mime="text/plain"
            )