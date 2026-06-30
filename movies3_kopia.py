import os
os.environ["PATH"] = r"C:\ProgramData\chocolatey\bin" + os.pathsep + os.environ["PATH"]

import streamlit as st
import tempfile
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
# Załadowanie zmiennych środowiskowych
load_dotenv()

class SubtitleGenerator:
    def __init__(self):
        self.client = None
        
    def setup_openai(self, api_key):
        """Konfiguracja OpenAI API"""
        try:
            self.client = OpenAI(api_key=api_key)
            #st.info(f"Używany klucz (początek): {api_key[:5]}***")
            return True
        except Exception as e:
            st.error(f"Błąd konfiguracji OpenAI: {e}")
            return False
    
    def extract_audio_from_video(self, video_path, audio_path):
        """Wyodrębnienie audio z pliku wideo"""
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return True

        except ffmpeg.Error as e:
            error_details = e.stderr.decode("utf8", errors="replace") if e.stderr else str(e)
            st.error(f"Szczegóły błędu FFmpeg:\n{error_details}")
            return False

        except Exception as e:
            st.error(f"Błąd wyodrębniania audio: {e}")
            return False
        
    def transcribe_audio(self, audio_path):
        """Transkrypcja audio na tekst za pomocą OpenAI Whisper"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            return transcript
        except Exception as e:
            st.error(f"Błąd transkrypcji: {e}")
            return None
    
    def create_srt_file(self, transcript, output_path):
        """Tworzenie pliku SRT z transkrypcji"""
        try:
            srt_content = ""
            for i, segment in enumerate(transcript.segments, 1):
                try:
                    # Próbuj jako obiekt (nowa wersja API)
                    start_time = self.seconds_to_srt_time(segment.start)
                    end_time = self.seconds_to_srt_time(segment.end)
                    text = segment.text.strip()
                except AttributeError:
                    # Jeśli nie działa, spróbuj jako słownik (stara wersja API)
                    start_time = self.seconds_to_srt_time(segment['start'])
                    end_time = self.seconds_to_srt_time(segment['end'])
                    text = segment['text'].strip()
                
                srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            return srt_content
        except Exception as e:
            st.error(f"Błąd tworzenia SRT: {e}")
            return None
    
    def seconds_to_srt_time(self, seconds):
        """Konwersja sekund do formatu czasowego SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def split_text_for_translation(self, subs, max_chunk_size=10):
        """Dzielenie napisów na fragmenty do tłumaczenia"""
        chunks = []
        current_chunk = []
        
        for sub in subs:
            current_chunk.append(sub)
            if len(current_chunk) >= max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def translate_srt(self, srt_content, target_language):
        """Tłumaczenie napisów na wybrany język używając GPT-4o"""
        try:
            # Mapowanie kodów języków na pełne nazwy
            language_names = {
                'en': 'angielski',
                'es': 'hiszpański', 
                'fr': 'francuski',
                'de': 'niemiecki',
                'it': 'włoski',
                'pl': 'polski',
                'ru': 'rosyjski',
                'ja': 'japoński',
                'ko': 'koreański',
                'zh': 'chiński'
            }
            
            target_lang_name = language_names.get(target_language, target_language)
            
            subs = pysrt.from_string(srt_content)
            translated_subs = []
            
            progress_bar = st.progress(0, text="Tłumaczenie napisów za pomocą GPT-4o...")
            total_subs = len(subs)
            
            # Grupowanie napisów w fragmenty dla efektywności
            chunk_size = 10  # Ile napisów na raz tłumaczyć
            chunks = self.split_text_for_translation(subs, chunk_size)
            
            for chunk_idx, sub_chunk in enumerate(chunks):
                # Połącz napisy z danego fragmentu w jeden tekst
                combined_text = "\n".join([f"[{sub.index}] {sub.text}" for sub in sub_chunk])
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system",
                             "content": f"""Jesteś profesjonalnym tłumaczem napisów filmowych. 
                             Przetłumacz poniższe napisy na język {target_lang_name}. 
                             
                             WAŻNE ZASADY:
                             - Zachowaj numerację [X] na początku każdej linii
                             - Tłumacz naturalnie, zachowując styl dialogu filmowego
                             - Nie dodawaj dodatkowych komentarzy
                             - Zachowaj emocjonalny ton wypowiedzi
                             - Dostosuj tłumaczenie do kontekstu kulturowego
                             - Jeśli to żart lub idiom, znajdź odpowiednik w docelowym języku
                             
                             Podaj tylko przetłumaczone teksty z zachowaną numeracją."""},
                            {"role": "user", "content": combined_text}
                        ],
                        temperature=0.3,
                        max_tokens=2000,
                    )
                    
                    translated_text = response.choices[0].message.content.strip()
                    translated_lines = translated_text.split('\n')
                    
                    # Przypisz przetłumaczone teksty do odpowiednich napisów
                    for i, sub in enumerate(sub_chunk):
                        if i < len(translated_lines):
                            # Usuń numerację [X] z przetłumaczonego tekstu
                            translated_line = translated_lines[i]
                            # Znajdź tekst po numeracji
                            if '] ' in translated_line:
                                translated_content = translated_line.split('] ', 1)[1]
                            else:
                                translated_content = translated_line
                            
                            translated_sub = pysrt.SubRipItem(
                                index=sub.index,
                                start=sub.start,
                                end=sub.end,
                                text=translated_content.strip()
                            )
                            translated_subs.append(translated_sub)
                        else:
                            # Fallback - zostaw oryginalny tekst
                            translated_subs.append(sub)
                    
                except Exception as e:
                    st.warning(f"⚠️ Błąd GPT-4o podczas tłumaczenia fragmentu {chunk_idx + 1}: {e}")
                    # Dodaj oryginalne napisy jako fallback
                    for sub in sub_chunk:
                        translated_subs.append(sub)
                
                # Aktualizuj progress bar
                progress_completed = min((chunk_idx + 1) * chunk_size, total_subs)
                progress_bar.progress(progress_completed / total_subs)
            
            progress_bar.empty()
            
            # POPRAWKA: Konwertuj obiekty na czytelny tekst SRT
            srt_output = ""
            for sub in translated_subs:
                srt_output += f"{sub.index}\n"
                srt_output += f"{sub.start} --> {sub.end}\n" 
                srt_output += f"{sub.text}\n\n"
            
            return srt_output  # Zwróć tekst, nie obiekty!
            
        except Exception as e:
            st.error(f"Błąd tłumaczenia: {e}")
            return None
    def embed_subtitles_to_video(self, video_path, srt_path, output_path, hard_subtitle=False):
        """Super prosta wersja - zawsze działa na Windows"""
        try:
            # Sprawdź czy pliki istnieją
            if not os.path.exists(video_path):
                st.error(f"Plik wideo nie istnieje: {video_path}")
                return False
            if not os.path.exists(srt_path):
                st.error(f"Plik napisów nie istnieje: {srt_path}")
                return False

            if hard_subtitle:
                # 🛡️ BULLETPROOF: Kopiuj napisy do tego samego folderu co wideo
                import shutil
                srt_name = "subs.srt"
                temp_srt = os.path.join(os.path.dirname(video_path), srt_name)
                shutil.copy2(srt_path, temp_srt)
            
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vf', f'subtitles={srt_name}',
                    '-c:a', 'copy',
                    '-y', output_path
                ]
            
                # Zmień working directory na folder z wideo
                old_cwd = os.getcwd()
                os.chdir(os.path.dirname(video_path))
            
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                finally:
                    # Zawsze wróć do oryginalnego folderu
                    os.chdir(old_cwd)
            else:
                # Miękkie napisy - te działają bez problemów
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', srt_path,
                    '-c', 'copy',
                    '-c:s', 'mov_text',
                    '-y', output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
            return True
        
        except subprocess.CalledProcessError as e:
            st.error(f"FFmpeg error: {e.stderr}")
            if "subtitles" in str(e.stderr).lower():
                st.info("💡 Spróbuj z miękkimi napisami - są bardziej uniwersalne")
            return False
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return False
    
    
    def get_video_duration(self, video_path):
        """Pobierz długość wideo w minutach"""
        try:
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe['streams'][0]['duration'])
            return duration / 60  # sekundy -> minuty
        except:
            # Fallback estimate based on file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            return file_size_mb * 0.3  # ~0.3 min per MB estimate

    def estimate_tokens(self, text, language='en'):
        """Szacuj liczbę tokenów (różne języki = różne tokeny)"""
        # Multipliers based on language
        multipliers = {
            'en': 1.0, 'es': 1.1, 'fr': 1.2, 'de': 1.3, 'it': 1.1,
            'pl': 1.4, 'ru': 1.6, 'ja': 0.7, 'ko': 0.8, 'zh': 0.6
        }
    
        char_count = len(text)
        base_tokens = char_count / 4  # 1 token ≈ 4 characters
        multiplier = multipliers.get(language, 1.0)
    
        return int(base_tokens * multiplier)

    def calculate_precise_costs(self, video_duration_minutes, srt_content, target_language, model_name):
        """Precyzyjne wyliczenie kosztów"""
    
        # 1. WHISPER COSTS (stałe)
        whisper_cost = video_duration_minutes * 0.006  # $0.006/min
    
        # 2. AI MODEL COSTS (różne dla każdego modelu)
        input_tokens = self.estimate_tokens(srt_content)
        output_tokens = int(input_tokens * 0.8)  # Tłumaczenie zwykle krótsze
    
        # Pricing per model (per 1M tokens)
        model_pricing = {
            'gpt-4o': {'input': 0.15, 'output': 0.60},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},  # Aktualne ceny, może się zmienić
            'claude-3.5': {'input': 0.30, 'output': 1.50},
            'gemini-pro': {'input': 0.125, 'output': 0.375},
            'groq-llama': {'input': 0.0, 'output': 0.0}  # Free tier
        }
    
        pricing = model_pricing.get(model_name, model_pricing['gpt-4o'])
    
        ai_input_cost = (input_tokens / 1_000_000) * pricing['input']
        ai_output_cost = (output_tokens / 1_000_000) * pricing['output'] 
        ai_total_cost = ai_input_cost + ai_output_cost
    
        # Porównanie z GPT-4o (baseline)
        gpt4o_pricing = model_pricing['gpt-4o']
        gpt4o_cost = (input_tokens / 1_000_000) * gpt4o_pricing['input'] + (output_tokens / 1_000_000) * gpt4o_pricing['output']
    
        savings_percent = (gpt4o_cost - ai_total_cost) / gpt4o_cost if gpt4o_cost > 0 else 0
        times_cheaper = gpt4o_cost / ai_total_cost if ai_total_cost > 0 else 1
    
        return {
            'whisper': whisper_cost,
            'ai_model': ai_total_cost,
            'total': whisper_cost + ai_total_cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'savings_percent': savings_percent,
            'times_cheaper': times_cheaper,
            'duration_mins': video_duration_minutes
        }    

def main():
    # ✅ DODAJ TĘ LINIĘ - ZWIĘKSZ LIMIT DO 1GB
    #st.set_option('server.maxUploadSize', 1000)
    
    st.set_page_config(
        page_title="Generator Napisów do Filmów - GPT-4o",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Generator Napisów do Filmów")
    st.markdown("*Potężne narzędzie z GPT-4o do tworzenia i tłumaczenia napisów, hmm!*")
    
    # Informacja o modelach
    st.info("🤖 **Używane modele AI:** OpenAI Whisper-1 (transkrypcja) + GPT-4o (tłumaczenie)")
    
    # Inicjalizacja generatora
    generator = SubtitleGenerator()
    
    # Sidebar z konfiguracją
    with st.sidebar:
        st.header("⚙️ Konfiguracja")
        
        # Klucz API OpenAI
        api_key = st.text_input(
            "Klucz API OpenAI", 
            type="password", 
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Wprowadź swój klucz API OpenAI (potrzebny do Whisper + GPT-4o)"
        )
        
        if api_key:
            if generator.setup_openai(api_key):
                st.success("✅ OpenAI API skonfigurowane")
                st.info("💰 Koszt: ~$0.006/min (Whisper) + ~$0.50-2.00 (GPT-4o tłumaczenie)")
            else:
                st.error("❌ Błąd konfiguracji OpenAI API")
        
        st.markdown("---")
        st.markdown("### 🎯 Model tłumaczenia")
        st.success("🚀 **GPT-4o** - Najlepszy dostępny model!")
        st.markdown("""
        **Zalety GPT-4o:**
        - 🎭 Rozumie kontekst filmowy
        - 🗣️ Naturalne dialogi 
        - 😄 Tłumaczy żarty i idiomy
        - 🎯 Spójność terminologii
        """)
    
    # Główna sekcja aplikacji
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Wczytaj plik wideo")
        uploaded_video = st.file_uploader(
            "Wybierz plik wideo", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maksymalny rozmiar: 500MB"  # ✅ ZAKTUALIZUJ TEKST
        )
        
        if uploaded_video:
            st.video(uploaded_video)
            # Informacje o pliku
            file_size = len(uploaded_video.read()) / (1024 * 1024)  # MB
            uploaded_video.seek(0)  # Reset pozycji
            st.caption(f"📊 Rozmiar pliku: {file_size:.1f} MB")

            # ✅ DODAJ WALIDACJĘ ROZMIARU
            MAX_FILE_SIZE = 1000  # MB - twój nowy limit
            if file_size > MAX_FILE_SIZE:
                st.error(f"🚫 Plik za duży! Maksymalny rozmiar: {MAX_FILE_SIZE}MB")
                st.info("💡 Spróbuj skompresować wideo lub użyć mniejszego pliku")
                st.stop()  # Zatrzymaj dalsze przetwarzanie
    
    with col2:
        st.header("🌍 Ustawienia tłumaczenia")
        
        languages = {
            'en': 'Angielski 🇺🇸',
            'es': 'Hiszpański 🇪🇸', 
            'fr': 'Francuski 🇫🇷',
            'de': 'Niemiecki 🇩🇪',
            'it': 'Włoski 🇮🇹',
            'pl': 'Polski 🇵🇱',
            'ru': 'Rosyjski 🇷🇺',
            'ja': 'Japoński 🇯🇵',
            'ko': 'Koreański 🇰🇷',
            'zh': 'Chiński 🇨🇳'
        }
        
        target_lang = st.selectbox(
            "Wybierz język tłumaczenia",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            help="GPT-4o przetłumaczy napisy na wybrany język"
        )
        
        subtitle_type = st.radio(
            "Typ napisów w wideo",
            ["Miękkie napisy (można wyłączyć)", "Twarde napisy (na stałe)"],
            help="Miękkie = można wyłączyć w odtwarzaczu, Twarde = wbudowane na stałe"
        )
        
        # Dodatkowe opcje
        with st.expander("🛠️ Opcje zaawansowane"):
            st.markdown("**Jakość tłumaczenia GPT-4o:**")
            st.success("🎯 Automatycznie dostosowana do napisów filmowych")
            st.markdown("- Temperatura: 0.3 (balans kreatywność/precyzja)")
            st.markdown("- Max tokens: 2000 na fragment") 
            st.markdown("- Fragmenty: 10 napisów na raz")
    
    # Główny przycisk przetwarzania
    if st.button("🚀 Generuj napisy z GPT-4o", type="primary", use_container_width=True):
        if not uploaded_video:
            st.error("❌ Wybierz plik wideo!")
            return
        
        if not api_key:
            st.error("❌ Wprowadź klucz API OpenAI!")
            return
        
        # Tworzenie tymczasowych plików
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Zapisanie wideo
            video_path = temp_dir / "input_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            # Ścieżki plików
            audio_path = temp_dir / "extracted_audio.wav"
            srt_original = temp_dir / "subtitles_original.srt"
            srt_translated = temp_dir / "subtitles_translated.srt"
            output_video = temp_dir / "output_with_subtitles.mp4"
            
            try:
                # Progress bar i status
                progress = st.progress(0)
                status_text = st.empty()
                
                # Krok 1: Wyodrębnianie audio
                status_text.text("🎵 Wyodrębnianie audio z wideo...")
                if generator.extract_audio_from_video(str(video_path), str(audio_path)):
                    progress.progress(15)
                    st.success("✅ Audio wyodrębnione")
                else:
                    st.error("❌ Błąd wyodrębniania audio")
                    return
                
                # Krok 2: Transkrypcja przez Whisper
                status_text.text("🎤 Transkrypcja audio przez OpenAI Whisper...")
                st.info("⏳ To może zająć kilka minut w zależności od długości filmu")
                transcript = generator.transcribe_audio(str(audio_path))
                if transcript:
                    progress.progress(40)
                    st.success("✅ Transkrypcja zakończona")
                else:
                    st.error("❌ Błąd transkrypcji")
                    return
                
                # Krok 3: Tworzenie SRT
                status_text.text("📝 Tworzenie pliku napisów SRT...")
                srt_content = generator.create_srt_file(transcript, str(srt_original))
                if srt_content:
                    progress.progress(55)
                    st.success("✅ Plik SRT utworzony")
                else:
                    st.error("❌ Błąd tworzenia SRT")
                    return
                
                # Krok 4: Tłumaczenie przez GPT-4o
                status_text.text(f"🤖 Tłumaczenie przez GPT-4o na {languages[target_lang]}...")
                st.info("🚀 GPT-4o analizuje kontekst i tłumaczy profesjonalnie...")
                translated_srt = generator.translate_srt(srt_content, target_lang)
                if translated_srt:
                    with open(srt_translated, 'w', encoding='utf-8') as f:
                        f.write(translated_srt)
                    progress.progress(80)
                    st.success("✅ Tłumaczenie GPT-4o zakończone")
                else:
                    st.error("❌ Błąd tłumaczenia GPT-4o")
                    return
                
                # Krok 5: Wtopianie napisów
                status_text.text("🎬 Wtopianie napisów do wideo...")
                hard_subs = subtitle_type == "Twarde napisy (na stałe)"
                
                if generator.embed_subtitles_to_video(
                    str(video_path), str(srt_translated), str(output_video), hard_subs
                ):
                    progress.progress(100)
                    status_text.text("✅ Napisy z GPT-4o gotowe!")
                    st.success("🎉 **Proces zakończony pomyślnie!**")
                else:
                    st.error("❌ Błąd wtopiania napisów")
                    return
                
                # Wyświetlanie wyników
                st.markdown("---")
                st.header("📋 Wyniki")
                
                # Podgląd napisów
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📄 Oryginalne napisy (Whisper)")
                    st.text_area("Transkrypcja z OpenAI Whisper", srt_content, height=300, key="original")
                
                with col2:
                    st.subheader(f"🌍 Napisy przetłumaczone (GPT-4o)")
                    st.caption(f"Język: {languages[target_lang]}")
                    st.text_area("Tłumaczenie z GPT-4o", translated_srt, height=300, key="translated")
                
                # Wideo z napisami
                st.subheader("🎬 Wideo z napisami")
                with open(output_video, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                
                # Pobieranie plików
                st.subheader("💾 Pobierz pliki")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📄 Pobierz oryginalne napisy (.srt)",
                        srt_content,
                        file_name="subtitles_original.srt",
                        mime="text/plain",
                        help="Napisy z OpenAI Whisper"
                    )
                
                with col2:
                    st.download_button(
                        f"🌍 Pobierz przetłumaczone napisy (.srt)",
                        translated_srt,
                        file_name=f"subtitles_{target_lang}_gpt4o.srt",
                        mime="text/plain",
                        help="Napisy przetłumaczone przez GPT-4o"
                    )
                
                with col3:
                    st.download_button(
                        "🎬 Pobierz wideo z napisami",
                        video_bytes,
                        file_name="video_with_subtitles_gpt4o.mp4",
                        mime="video/mp4",
                        help="Film z wtopionych napisami GPT-4o"
                    )
                
                # Podsumowanie kosztów
                st.markdown("---")
                with st.expander("💰 Szacowane koszty sesji"):
                    st.markdown(f"""
                    **Użyte modele:**
                    - 🎤 **Whisper-1**: ~${file_size * 0.1:.2f} (transkrypcja audio)
                    - 🤖 **GPT-4o**: ~$0.50-2.00 (tłumaczenie napisów)
                    
                    **Całkowity szacowany koszt**: ~${file_size * 0.1 + 1:.2f}
                    """)
                
            except Exception as e:
                st.error(f"❌ Błąd podczas przetwarzania: {str(e)}")
                st.markdown("🔧 **Porady rozwiązywania problemów:**")
                st.markdown("- Sprawdź połączenie internetowe")
                st.markdown("- Upewnij się, że klucz API jest prawidłowy")
                st.markdown("- Spróbuj z mniejszym plikiem wideo")

if __name__ == "__main__":
    main()
