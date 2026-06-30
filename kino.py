import streamlit as st
import tempfile
import os
import subprocess
from pathlib import Path
import requests
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pydub import AudioSegment
import pysrt
import ffmpeg
import zipfile
from datetime import timedelta
import re
import base64

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja strony
st.set_page_config(
    page_title="Generator Napisów do Filmów",
    page_icon="🎬",
    layout="wide"
)

class SubtitleGenerator:
    def __init__(self):
        self.client = None
        self.setup_openai()
    
    def setup_openai(self):
        """Konfiguracja klienta OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Brak klucza API OpenAI! Dodaj go do pliku .env")
            return False
        
        try:
            # Poprawiona konfiguracja - bez argumentu 'proxies'
            self.client = OpenAI(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"❌ Błąd konfiguracji OpenAI: {str(e)}")
            return False
    
    def extract_audio_from_video(self, video_path, output_path):
        """Wyodrębnia dźwięk z wideo - jak wyciąganie miąższu z pomarańczy"""
        try:
            # Używamy ffmpeg do ekstraktowania audio
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            return True
        except ffmpeg.Error as e:
            st.error(f"❌ Błąd podczas wyodrębniania dźwięku: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Błąd ogólny przy ekstraktowaniu audio: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transkrypcja audio na tekst - jak tłumacz który słucha i pisze"""
        if not self.client:
            return None
        
        try:
            with open(audio_path, "rb") as audio_file:
                # Używamy modelu Whisper do transkrypcji
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            return transcript
        except Exception as e:
            st.error(f"❌ Błąd podczas transkrypcji: {str(e)}")
            return None
    
    def extract_audio_from_video(self, video_path, output_path):
        """Wyodrębnia dźwięk z wideo - jak wyciąganie miąższu z pomarańczy"""
        try:
            # Używamy ffmpeg do ekstraktowania audio
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            return True
        except ffmpeg.Error as e:
            st.error(f"❌ Błąd podczas wyodrębniania dźwięku: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Błąd ogólny przy ekstraktowaniu audio: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transkrypcja audio na tekst - jak tłumacz który słucha i pisze"""
        if not self.client:
            return None
        
        try:
            with open(audio_path, "rb") as audio_file:
                # Używamy modelu Whisper do transkrypcji
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            return transcript
        except Exception as e:
            st.error(f"❌ Błąd podczas transkrypcji: {str(e)}")
            return None
    
    def create_srt_from_transcript(self, transcript, srt_path):
        """Tworzy plik SRT z napisami - jak układanie puzzli w czasie"""
        try:
            subtitles = []
            subtitle_index = 1
        
            # Sprawdzamy czy mamy dostęp do słów z timestampami
            if hasattr(transcript, 'words') and transcript.words:
                # Grupujemy słowa w segmenty po ~5-7 słów
                words = transcript.words
                segment_words = []
                segment_duration = 3.0  # 3 sekundy na segment
            
                for i, word in enumerate(words):
                    segment_words.append(word)
                
                    # Tworzymy nowy segment co kilka słów lub na końcu
                    if len(segment_words) >= 6 or i == len(words) - 1:
                        if segment_words:
                            start_time = segment_words[0].start  # ✅ Użyj .start zamiast ['start']
                            end_time = min(
                                segment_words[-1].end,  # ✅ Użyj .end zamiast ['end']
                                start_time + segment_duration
                            )
                        
                            # Łączymy słowa w tekst
                            text = ' '.join([w.word for w in segment_words])  # ✅ Użyj .word zamiast ['word']
                        
                            # Tworzymy subtitle
                            subtitle = pysrt.SubRipItem(
                                index=subtitle_index,
                                start=pysrt.SubRipTime(seconds=start_time),
                                end=pysrt.SubRipTime(seconds=end_time),
                                text=text
                            )
                        
                            subtitles.append(subtitle)
                            subtitle_index += 1
                            segment_words = []
        
            else:
                # Fallback - jeśli nie mamy timestampów słów, używamy całego tekstu
                text = transcript.text
                duration = 5.0  # 5 sekund na segment
                words_per_segment = 8
                words = text.split()
            
                for i in range(0, len(words), words_per_segment):
                    segment_words = words[i:i + words_per_segment]
                    segment_text = ' '.join(segment_words)
                    start_time = i * duration / words_per_segment
                    end_time = start_time + duration
                
                    subtitle = pysrt.SubRipItem(
                        index=subtitle_index,
                        start=pysrt.SubRipTime(seconds=start_time),
                        end=pysrt.SubRipTime(seconds=end_time),
                        text=segment_text
                    )
                
                    subtitles.append(subtitle)
                    subtitle_index += 1
        
            # Zapisujemy do pliku SRT
            srt_file = pysrt.SubRipFile(subtitles)
            srt_file.save(srt_path, encoding='utf-8')
        
            return True
        
        except Exception as e:
            st.error(f"❌ Błąd podczas tworzenia napisów: {str(e)}")
            return False
    
    
    def convert_srt_to_vtt(self, srt_content):
        """Konwertuje SRT do WebVTT - jak tłumaczenie między dialektami"""
        vtt_content = "WEBVTT\n\n"
        
        # Zamieniamy format czasu z SRT na VTT
        lines = srt_content.split('\n')
        for line in lines:
            if '-->' in line:
                # Zamieniamy przecinek na kropkę w czasie
                line = line.replace(',', '.')
            vtt_content += line + '\n'
        
        return vtt_content

def main():
    st.title("🎬 Generator Napisów do Filmów")
    st.markdown("### Zamień swoje wideo w film z napisami - jak dodawanie głosu do niemego kina! 🎭")
    
    # Inicjalizacja generatora
    generator = SubtitleGenerator()
    
    if not generator.client:
        st.warning("⚠️ Skonfiguruj najpierw klucz API OpenAI w pliku .env")
        st.code("OPENAI_API_KEY=twój_klucz_api_tutaj")
        st.info("💡 Utwórz plik `.env` w folderze z aplikacją i dodaj swój klucz API OpenAI")
        return
    
    # Sidebar z informacjami
    with st.sidebar:
        st.header("📋 Instrukcja")
        st.markdown("""
        **Krok po kroku:**
        1. 📁 Załaduj plik wideo
        2. ⚡ Kliknij "Generuj Napisy"
        3. ⏳ Poczekaj na magię
        4. 🎉 Oglądaj z napisami!
        
        **Obsługiwane formaty:**
        - MP4, AVI, MOV, MKV
        - Maksymalnie 200MB
        
        **Wymagania:**
        - Klucz API OpenAI w pliku .env
        - Dobra jakość dźwięku w wideo
        """)
    
    # Upload pliku wideo
    uploaded_file = st.file_uploader(
        "📁 Wybierz plik wideo",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Przeciągnij i upuść swój film tutaj - jak wrzucanie listu do skrzynki!",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Sprawdzamy rozmiar pliku
        file_size_mb = uploaded_file.size / 1024 / 1024
        if file_size_mb > 200:
            st.error("❌ Plik jest za duży! Maksymalny rozmiar to 200MB")
            return
        
        # Wyświetlamy informacje o pliku
        st.success(f"✅ Załadowano: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Podgląd wideo
        st.subheader("📺 Podgląd wideo")
        st.video(uploaded_file)
        
        # Przycisk do generowania napisów
        if st.button("🚀 Generuj Napisy", type="primary"):
            
            # Progress bar - jak pasek ładowania w grze
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Zapisujemy wideo
                    status_text.text("📁 Zapisywanie wideo...")
                    progress_bar.progress(10)
                    
                    video_path = os.path.join(temp_dir, f"video_{uploaded_file.name}")
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Wyodrębniamy audio
                    status_text.text("🎵 Wyodrębnianie dźwięku...")
                    progress_bar.progress(30)
                    
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    if not generator.extract_audio_from_video(video_path, audio_path):
                        st.error("❌ Nie udało się wyodrębnić dźwięku. Sprawdź format pliku.")
                        return
                    
                    # Sprawdzamy czy plik audio został utworzony
                    if not os.path.exists(audio_path):
                        st.error("❌ Plik audio nie został utworzony")
                        return
                    
                    # Transkrypcja
                    status_text.text("🤖 Analizowanie mowy... (to może potrwać kilka minut)")
                    progress_bar.progress(60)
                    
                    transcript = generator.transcribe_audio(audio_path)
                    if not transcript:
                        st.error("❌ Nie udało się przeprowadzić transkrypcji. Sprawdź jakość dźwięku.")
                        return
                    
                    # Tworzenie napisów
                    status_text.text("📝 Tworzenie napisów...")
                    progress_bar.progress(80)
                    
                    srt_path = os.path.join(temp_dir, "subtitles.srt")
                    if not generator.create_srt_from_transcript(transcript, srt_path):
                        st.error("❌ Nie udało się utworzyć napisów")
                        return
                    
                    # Finalizacja
                    status_text.text("🎬 Przygotowywanie rezultatu...")
                    progress_bar.progress(100)
                    
                    # Wyświetlamy rezultat
                    st.success("🎉 Napisy zostały wygenerowane!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📺 Wideo z napisami")
                        st.video(uploaded_file)
                        st.info("💡 Aby zobaczyć napisy, pobierz plik SRT i użyj odtwarzacza obsługującego napisy (VLC, inne)")
                    
                    with col2:
                        st.subheader("📜 Wygenerowane napisy")
                        with open(srt_path, "r", encoding='utf-8') as f:
                            subtitles_content = f.read()
                        
                        # Pokazujemy fragment napisów
                        st.text_area(
                            "Podgląd napisów:",
                            subtitles_content[:1000] + "\n..." if len(subtitles_content) > 1000 else subtitles_content,
                            height=300
                        )
                        
                        # Przycisk do pobrania
                        st.download_button(
                            label="📥 Pobierz napisy (SRT)",
                            data=subtitles_content,
                            file_name=f"napisy_{uploaded_file.name}.srt",
                            mime="text/plain"
                        )
                    
                    # Informacje o transkrypcji
                    with st.expander("📊 Statystyki transkrypcji"):
                        if hasattr(transcript, 'language'):
                            st.write(f"**Język wykryty:** {transcript.language}")
                        if hasattr(transcript, 'words'):
                            st.write(f"**Liczba słów:** {len(transcript.words)}")
                        st.write(f"**Pełny tekst:**")
                        st.text_area("Transkrypcja:", transcript.text, height=150)
                    
                    status_text.text("✅ Gotowe!")
                    
            except Exception as e:
                st.error(f"❌ Wystąpił błąd: {str(e)}")
                st.error("💡 Sprawdź czy plik nie jest uszkodzony i spróbuj ponownie")

if __name__ == "__main__":
    main()