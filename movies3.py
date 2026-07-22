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
    def get_detected_language(self, transcript):
        """Zwraca kod języka wykryty przez Whisper, np. pl, en, es."""
        try:
            if transcript is None:
                st.error("Brak wyniku transkrypcji — nie można wykryć języka.")
                return None

            language = getattr(transcript, "language", None)

            if language is None and isinstance(transcript, dict):
                language = transcript.get("language")

            if not language or not isinstance(language, str):
                st.warning("Whisper nie zwrócił kodu wykrytego języka.")
                return None

            return language.lower()

        except Exception as e:
            st.error(f"Błąd odczytu języka z transkrypcji: {e}")
            return None

    def get_supported_languages(self):
        """Zwraca słownik obsługiwanych języków: kod -> nazwa przyjazna użytkownikowi."""
        return {
            'en': 'Angielski 🇺🇸/🇬🇧',
            'es': 'Hiszpański 🇪🇸',
            'fr': 'Francuski 🇫🇷',
            'de': 'Niemiecki 🇩🇪',
            'it': 'Włoski 🇮🇹',
            'pl': 'Polski 🇵🇱',
            'ru': 'Rosyjski 🇷🇺',
            'ja': 'Japoński 🇯🇵',
            'ko': 'Koreański 🇰🇷',
            'zh': 'Chiński 🇨🇳',
            'pt': 'Portugalski 🇵🇹/🇧🇷',
            'nl': 'Holenderski 🇳🇱',
            'ar': 'Arabski 🇸🇦',
            'tr': 'Turecki 🇹🇷',
            'uk': 'Ukraiński 🇺🇦',
            'cs': 'Czeski 🇨🇿',
            'sk': 'Słowacki 🇸🇰',
            'el': 'Grecki 🇬🇷',
            'hu': 'Węgierski 🇭🇺',
            'sv': 'Szwedzki 🇸🇪',
            'da': 'Duński 🇩🇰',
            'fi': 'Fiński 🇫🇮',
            'no': 'Norweski 🇳🇴',
            'ro': 'Rumuński 🇷🇴',
            'bg': 'Bułgarski 🇧🇬',
            'hr': 'Chorwacki 🇭🇷',
            'sr': 'Serbski 🇷🇸',
            'th': 'Tajski 🇹🇭',
            'vi': 'Wietnamski 🇻🇳',
            'id': 'Indonezyjski 🇮🇩',
            'he': 'Hebrajski 🇮🇱',
            'hi': 'Hindi 🇮🇳'
        }

    def get_language_name(self, language_code):
        """Zwraca przyjazną nazwę języka na podstawie kodu."""
        languages = self.get_supported_languages()

        if not language_code:
            return "Nieznany język"

        return languages.get(language_code.lower(), "Nieznany język")

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

                srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            return srt_content

        except Exception as e:
            st.error(f"Błąd tworzenia SRT: {e}")
            return None
     

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

        languages = generator.get_supported_languages()

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
                    detected_language = generator.get_detected_language(transcript)

                    progress.progress(40)
                    st.success("✅ Transkrypcja zakończona")

                    if detected_language:
                         detected_language_name = generator.get_language_name(detected_language)

                         st.success(
                            f"🌍 Wykryty język filmu: "
                            f"{detected_language_name} ({detected_language})"
                        )
                    else:
                        st.warning("⚠️ Nie udało się automatycznie wykryć języka filmu.")
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