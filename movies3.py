import streamlit as st
import tempfile
import os
import subprocess
from pathlib import Path
import requests
import openai
from openai import OpenAI
from pydub import AudioSegment
import pysrt
import ffmpeg
import zipfile
from datetime import timedelta
import re

class SubtitleGenerator:

    def __init__(self):
        self.client = None

    def setup_openai(self, api_key):
        """Konfiguracja OpenAI API"""
        try:
            self.client = OpenAI(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Błąd konfiguracji OpenAI: {e}")
            return False

    def extract_audio_from_video(self, video_path, audio_path):
        """Wyodrębnienie audio z pliku wideo"""
        try:
            ffmpeg.input(video_path).output(audio_path, acodec='libmp3lame', ar=16000).overwrite_output().run(quiet=True)
            return True
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
                    start_time = self.seconds_to_srt_time(segment.start)
                    end_time = self.seconds_to_srt_time(segment.end)
                    text = segment.text.strip()
                except AttributeError:
                    start_time = self.seconds_to_srt_time(segment['start'])
                    end_time = self.seconds_to_srt_time(segment['end'])
                    text = segment['text'].strip()
                # ✅ POPRAWIONE: HTML entity na normalny znak
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

    def get_video_duration(self, video_path):
        """Pobierz długość wideo w minutach"""
        try:
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe['streams'][0]['duration'])
            return duration / 60
        except Exception as e:
            # Fallback estimate based on file size (więcej szczegółów w error handling)
            st.warning(f"Nie można odczytać długości wideo: {e}")
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            estimated_duration = file_size_mb * 0.3  # ~0.3 min per MB estimate
            st.info(f"Szacowana długość na podstawie rozmiaru pliku: {estimated_duration:.1f} min")
            return estimated_duration

    def estimate_tokens(self, text, language='en'):
        """Szacuj liczbę tokenów (różne języki = różne tokeny)"""
        multipliers = {
            'en': 1.0, 'es': 1.1, 'fr': 1.2, 'de': 1.3, 'it': 1.1,
            'pl': 1.4, 'ru': 1.6, 'ja': 0.7, 'ko': 0.8, 'zh': 0.6
        }
        char_count = len(text)
        base_tokens = char_count / 4  # 1 token ≈ 4 characters
        multiplier = multipliers.get(language, 1.0)
        return int(base_tokens * multiplier)

    def calculate_precise_costs(self, video_duration_minutes, srt_content, target_language, model_name):
        """✅ POPRAWIONE: Precyzyjne wyliczenie kosztów z prawdziwymi cenami"""
        
        # 1. WHISPER COSTS (stałe)
        whisper_cost = video_duration_minutes * 0.006  # $0.006/min
        
        # 2. AI MODEL COSTS - PRAWDZIWE CENY (wrzesień 2025)
        input_tokens = self.estimate_tokens(srt_content, target_language)
        output_tokens = int(input_tokens * 0.8)  # Tłumaczenie zwykle krótsze
        
        # ✅ PRAWDZIWE CENY OPENAI (per 1M tokens)
        model_pricing = {
            'gpt-4o': {
                'input': 5.00,    # $5.00/1M input tokens
                'output': 15.00,  # $15.00/1M output tokens
                'display_name': 'GPT-4o'
            },
            'gpt-4o-mini': {
                'input': 0.15,    # $0.15/1M input tokens (33x tańsze input)
                'output': 0.60,   # $0.60/1M output tokens (25x tańsze output)
                'display_name': 'GPT-4o-mini'
            }
        }
        
        pricing = model_pricing.get(model_name, model_pricing['gpt-4o'])
        
        # Wylicz koszty AI
        ai_input_cost = (input_tokens / 1_000_000) * pricing['input']
        ai_output_cost = (output_tokens / 1_000_000) * pricing['output']
        ai_total_cost = ai_input_cost + ai_output_cost
        
        # Porównanie z GPT-4o (baseline)
        gpt4o_pricing = model_pricing['gpt-4o']
        gpt4o_input_cost = (input_tokens / 1_000_000) * gpt4o_pricing['input']
        gpt4o_output_cost = (output_tokens / 1_000_000) * gpt4o_pricing['output']
        gpt4o_total_cost = gpt4o_input_cost + gpt4o_output_cost
        
        # Wylicz oszczędności
        savings_percent = (gpt4o_total_cost - ai_total_cost) / gpt4o_total_cost if gpt4o_total_cost > 0 else 0
        times_cheaper = gpt4o_total_cost / ai_total_cost if ai_total_cost > 0 else 1
        
        return {
            'whisper': whisper_cost,
            'ai_model': ai_total_cost,
            'ai_input_cost': ai_input_cost,
            'ai_output_cost': ai_output_cost,
            'total': whisper_cost + ai_total_cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'savings_percent': savings_percent,
            'times_cheaper': times_cheaper,
            'duration_mins': video_duration_minutes,
            'model_display_name': pricing['display_name'],
            'input_price_per_1m': pricing['input'],
            'output_price_per_1m': pricing['output']
        }

    def translate_srt_with_model(self, srt_content, target_language, model_name):
        """Universal translation method with model choice"""
        try:
            language_names = {
                'en': 'angielski', 'es': 'hiszpański', 'fr': 'francuski',
                'de': 'niemiecki', 'it': 'włoski', 'pl': 'polski',
                'ru': 'rosyjski', 'ja': 'japoński', 'ko': 'koreański', 'zh': 'chiński'
            }
            target_lang_name = language_names.get(target_language, target_language)
            subs = pysrt.from_string(srt_content)
            translated_subs = []
            
            model_display = "GPT-4o" if model_name == "gpt-4o" else "GPT-4o-mini"
            progress_bar = st.progress(0, text=f"Tłumaczenie napisów za pomocą {model_display}...")
            total_subs = len(subs)
            chunk_size = 10
            chunks = self.split_text_for_translation(subs, chunk_size)
            
            for chunk_idx, sub_chunk in enumerate(chunks):
                combined_text = "\n".join([f"[{sub.index}] {sub.text}" for sub in sub_chunk])
                try:
                    response = self.client.chat.completions.create(
                        model=model_name,  # ✅ DYNAMIC MODEL
                        messages=[
                            {"role": "system", "content": f"""Jesteś profesjonalnym tłumaczem napisów filmowych.
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
                    
                    for i, sub in enumerate(sub_chunk):
                        if i < len(translated_lines):
                            translated_line = translated_lines[i]
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
                            translated_subs.append(sub)
                except Exception as e:
                    st.warning(f"⚠️ Błąd {model_display} podczas tłumaczenia fragmentu {chunk_idx + 1}: {e}")
                    for sub in sub_chunk:
                        translated_subs.append(sub)
                
                progress_completed = min((chunk_idx + 1) * chunk_size, total_subs)
                progress_bar.progress(progress_completed / total_subs)
            
            progress_bar.empty()
            
            # ✅ POPRAWIONE: HTML entity na normalny znak
            srt_output = ""
            for sub in translated_subs:
                srt_output += f"{sub.index}\n{sub.start} --> {sub.end}\n{sub.text}\n\n"
            return srt_output
        except Exception as e:
            st.error(f"Błąd tłumaczenia: {e}")
            return None

    def embed_subtitles_to_video(self, video_path, srt_path, output_path, hard_subtitle=False):
        """Super prosta wersja - zawsze działa na Windows"""
        try:
            if not os.path.exists(video_path):
                st.error(f"Plik wideo nie istnieje: {video_path}")
                return False
            if not os.path.exists(srt_path):
                st.error(f"Plik napisów nie istnieje: {srt_path}")
                return False
            if hard_subtitle:
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
                old_cwd = os.getcwd()
                os.chdir(os.path.dirname(video_path))
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                finally:
                    os.chdir(old_cwd)
            else:
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


def main():
    st.set_page_config(
        page_title="Generator Napisów do Filmów - AI",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Generator Napisów do Filmów")
    st.markdown("*Potężne narzędzie z AI do tworzenia i tłumaczenia napisów!*")

    # ✅ ZAKTUALIZOWANE: Informacja o modelach
    st.info("🤖 **Używane modele AI:** OpenAI Whisper-1 (transkrypcja) + GPT-4o/GPT-4o-mini (tłumaczenie)")
    st.warning("🔑 **Potrzebujesz własnego klucza API OpenAI**")

    with st.expander("📖 Jak uzyskać klucz API OpenAI?"):
        st.markdown("""
        1. Idź na [platform.openai.com](https://platform.openai.com/)
        2. Zaloguj się lub stwórz konto
        3. Przejdź do **API Keys** w menu
        4. Kliknij **Create new secret key**
        5. Skopiuj klucz i wklej go w polu obok ⬅️
        
        💰 **Koszt**: $0.006/min (Whisper) + $0.15-15.00/1M tokens (AI)
        """)

    generator = SubtitleGenerator()

    with st.sidebar:
        st.header("⚙️ Konfiguracja")
        api_key = st.text_input(
            "Klucz API OpenAI",
            type="password",
            placeholder="sk-proj-...",
            help="Wprowadź swój klucz API OpenAI"
        )

        if api_key:
            if not api_key.startswith('sk-'):
                st.error("❌ Nieprawidłowy format klucza! Powinien zaczynać się od 'sk-'")
            elif generator.setup_openai(api_key):
                st.success("✅ OpenAI API skonfigurowane")
            else:
                st.error("❌ Błąd konfiguracji OpenAI API - sprawdź klucz")
        else:
            st.info("👈 Wprowadź klucz API w panelu z lewej strony")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📹 Wczytaj plik wideo")
        uploaded_video = st.file_uploader(
            "Wybierz plik wideo",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maksymalny rozmiar: 1GB"
        )

        if uploaded_video:
            st.video(uploaded_video)
            file_size = len(uploaded_video.read()) / (1024 * 1024)  # MB
            uploaded_video.seek(0)
            st.caption(f"📊 Rozmiar pliku: {file_size:.1f} MB")

            MAX_FILE_SIZE = 1000  # MB
            if file_size > MAX_FILE_SIZE:
                st.error(f"🚫 Plik za duży! Maksymalny rozmiar: {MAX_FILE_SIZE}MB")
                st.info("💡 Spróbuj skompresować wideo lub użyć mniejszego pliku")
                st.stop()

    with col2:
        st.header("🌍 Ustawienia tłumaczenia")
        
        st.subheader("🧠 Model AI")
        # ✅ ZAKTUALIZOWANE: Prawdziwe informacje o kosztach
        translation_models = {
            'gpt-4o-mini': {
                'name': 'GPT-4o Mini 💸',
                'price': '$',
                'quality': '⭐⭐⭐⭐',
                'speed': '🚀 Szybszy',
                'description': 'Bardzo dobra jakość, około 33x tańszy input, 25x tańszy output'
            },
            'gpt-4o': {
                'name': 'GPT-4o 🏆',
                'price': '$$$$$',
                'quality': '⭐⭐⭐⭐⭐',
                'speed': '🐢 Wolniejszy',
                'description': 'Najwyższa jakość, rozumie kontekst filmowy'
            }
        }

        selected_model = st.selectbox(
            "Wybierz model tłumaczenia",
            options=list(translation_models.keys()),
            format_func=lambda x: translation_models[x]['name'],
            help="Różne modele = różne ceny i jakość",
            index=0  # Domyślnie GPT-4o-mini (tańszy)
        )

        model_info = translation_models[selected_model]
        st.info(f"""
**{model_info['name']}**
💰 Koszt: {model_info['price']} | 🎯 Jakość: {model_info['quality']} | ⚡ Szybkość: {model_info['speed']}

{model_info['description']}
""")

        languages = {
            'en': 'Angielski 🇺🇸', 'es': 'Hiszpański 🇪🇸', 'fr': 'Francuski 🇫🇷',
            'de': 'Niemiecki 🇩🇪', 'it': 'Włoski 🇮🇹', 'pl': 'Polski 🇵🇱',
            'ru': 'Rosyjski 🇷🇺', 'ja': 'Japoński 🇯🇵', 'ko': 'Koreański 🇰🇷',
            'zh': 'Chiński 🇨🇳'
        }

        target_lang = st.selectbox(
            "Wybierz język tłumaczenia",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            help="AI przetłumaczy napisy na wybrany język"
        )

        subtitle_type = st.radio(
            "Typ napisów w wideo",
            ["Miękkie napisy (można wyłączyć)", "Twarde napisy (na stałe)"],
            help="Miękkie = można wyłączyć w odtwarzaczu, Twarde = wbudowane na stałe"
        )

    if st.button("🚀 Generuj napisy z AI", type="primary", use_container_width=True):
        if not uploaded_video:
            st.error("❌ Wybierz plik wideo!")
            return
        if not api_key:
            st.error("❌ Wprowadź klucz API OpenAI!")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            video_path = temp_dir / "input_video.mp4"

            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            audio_path = temp_dir / "extracted_audio.mp3"
            srt_original = temp_dir / "subtitles_original.srt"
            srt_translated = temp_dir / "subtitles_translated.srt"
            output_video = temp_dir / "output_with_subtitles.mp4"

            try:
                progress = st.progress(0)
                status_text = st.empty()

                status_text.text("🎵 Wyodrębnianie audio z wideo...")
                if generator.extract_audio_from_video(str(video_path), str(audio_path)):
                    progress.progress(15)
                    st.success("✅ Audio wyodrębnione")
                else:
                    st.error("❌ Błąd wyodrębniania audio")
                    return

                status_text.text("🎤 Transkrypcja audio przez OpenAI Whisper...")
                st.info("⏳ To może zająć kilka minut w zależności od długości filmu")
                transcript = generator.transcribe_audio(str(audio_path))
                if transcript:
                    progress.progress(40)
                    st.success("✅ Transkrypcja zakończona")
                else:
                    st.error("❌ Błąd transkrypcji")
                    return

                status_text.text("📝 Tworzenie pliku napisów SRT...")
                srt_content = generator.create_srt_file(transcript, str(srt_original))
                if srt_content:
                    progress.progress(55)
                    st.success("✅ Plik SRT utworzony")
                else:
                    st.error("❌ Błąd tworzenia SRT")
                    return

                model_display = translation_models[selected_model]['name']
                status_text.text(f"🤖 Tłumaczenie przez {model_display} na {languages[target_lang]}...")
                st.info(f"🚀 {model_display} analizuje kontekst i tłumaczy profesjonalnie...")
                
                translated_srt = generator.translate_srt_with_model(srt_content, target_lang, selected_model)
                if translated_srt:
                    with open(srt_translated, 'w', encoding='utf-8') as f:
                        f.write(translated_srt)
                    progress.progress(80)
                    st.success(f"✅ Tłumaczenie {model_display} zakończone")
                else:
                    st.error("❌ Błąd tłumaczenia AI")
                    return

                status_text.text("🎬 Wtopianie napisów do wideo...")
                hard_subs = subtitle_type == "Twarde napisy (na stałe)"
                if generator.embed_subtitles_to_video(str(video_path), str(srt_translated), str(output_video), hard_subs):
                    progress.progress(100)
                    status_text.text("✅ Napisy z AI gotowe!")
                    st.success("🎉 **Proces zakończony pomyślnie!**")
                else:
                    st.error("❌ Błąd wtopiania napisów")
                    return

                st.markdown("---")
                st.header("📋 Wyniki")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 Oryginalne napisy (Whisper)")
                    st.text_area("Transkrypcja z OpenAI Whisper", srt_content, height=300, key="original")

                with col2:
                    st.subheader(f"🌍 Napisy przetłumaczone ({model_display})")
                    st.caption(f"Język: {languages[target_lang]} | Model: {selected_model}")
                    st.text_area("Tłumaczenie AI", translated_srt, height=300, key="translated")

                st.subheader("🎬 Wideo z napisami")
                with open(output_video, "rb") as f:
                    video_bytes = f.read()
                    st.video(video_bytes)

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
                        file_name=f"subtitles_{target_lang}_{selected_model}.srt",
                        mime="text/plain",
                        help="Napisy przetłumaczone przez AI"
                    )

                with col3:
                    st.download_button(
                        "🎬 Pobierz wideo z napisami",
                        video_bytes,
                        file_name="video_with_subtitles.mp4",
                        mime="video/mp4",
                        help="Film z wtopionymi napisami AI"
                    )

                # ✅ NOWA PRECYZYJNA KALKULACJA KOSZTÓW
                video_duration = generator.get_video_duration(str(video_path))
                costs = generator.calculate_precise_costs(video_duration, srt_content, target_lang, selected_model)

                st.markdown("---")
                with st.expander("💰 Szczegółowy breakdown kosztów"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🎤 Whisper-1", f"${costs['whisper']:.4f}", f"{costs['duration_mins']:.1f} min × $0.006")
                        st.metric("🧠 Input Tokens", f"${costs['ai_input_cost']:.4f}", 
                                 f"{costs['input_tokens']:,} × ${costs['input_price_per_1m']}/1M")
                        st.metric("🎯 Output Tokens", f"${costs['ai_output_cost']:.4f}", 
                                 f"{costs['output_tokens']:,} × ${costs['output_price_per_1m']}/1M")
                    with col_b:
                        st.metric("💵 Model AI Całość", f"${costs['ai_model']:.4f}", 
                                 f"{costs['model_display_name']}")
                        st.metric("🏆 KOSZT CAŁKOWITY", f"${costs['total']:.4f}", 
                                 f"Whisper + {costs['model_display_name']}")
                        if selected_model != 'gpt-4o' and costs['times_cheaper'] > 1:
                            st.success(f"✅ Oszczędność vs GPT-4o: {costs['savings_percent']:.1%}")
                            st.info(f"🎯 GPT-4o-mini jest {costs['times_cheaper']:.0f}x tańszy!")

            except Exception as e:
                st.error(f"❌ Błąd podczas przetwarzania: {str(e)}")
                st.markdown("🔧 **Porady rozwiązywania problemów:**")
                st.markdown("- Sprawdź połączenie internetowe")
                st.markdown("- Upewnij się, że klucz API jest prawidłowy")
                st.markdown("- Spróbuj z mniejszym plikiem wideo")


if __name__ == "__main__":
    main()