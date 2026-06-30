import os
import re
import json
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values
from pydub import AudioSegment
from openai import OpenAI

# ===========================
# Konfiguracja i helpery
# ===========================

env = dotenv_values(".env")
API_KEY = env.get("OPENAI_API_KEY")

if not API_KEY:
    st.error("❌ Nie znaleziono OPENAI_API_KEY w .env")
    st.stop()

def get_openai_client():
    return OpenAI(api_key=API_KEY)

SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mp3", "wav", "m4a"]
MAX_SIZE = 500 * 1024 * 1024  # 500MB
CHUNK_LENGTH_MS = 4 * 60 * 1000  # 4 min

def srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt(segments: list) -> str:
    srt_content = []
    for i, seg in enumerate(segments, 1):
        start = srt_timestamp(seg.get('start', 0))
        end = srt_timestamp(seg.get('end', 0))
        text = re.sub(r'\s+', ' ', seg.get('text', '')).strip()
        if text:
            srt_content.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_content)

def split_text(text, max_len=3000):
    sentences = text.replace("\n", " ").split('. ')
    chunks = []
    current = ""
    for sent in sentences:
        s = sent.strip()
        if s and not s.endswith('.'):
            s += '.'
        s += " "
        if len(current) + len(s) <= max_len:
            current += s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks

def translate_long_text(client, text, target_lang):
    chunks = split_text(text, max_len=3000)
    translations = []
    progress_bar = st.progress(0, text="Tłumaczenie w toku...")
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": f"Jesteś profesjonalnym tłumaczem. Przetłumacz poniższy tekst na język {target_lang}. Podaj tylko przetłumaczony tekst, bez dodatkowych komentarzy."},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3,
            )
            translated = response.choices[0].message.content.strip()
            translations.append(translated)
        except Exception as e:
            st.warning(f"⚠️ Błąd podczas tłumaczenia fragmentu {idx + 1}: {e}")
            translations.append("[BŁĄD TŁUMACZENIA]")
        progress_bar.progress((idx + 1) / total)
    progress_bar.empty()
    return "\n\n".join(translations)

def split_audio(file_path, chunk_length_ms=CHUNK_LENGTH_MS):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"{file_path}_chunk_{i // chunk_length_ms}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

def srt_to_ass(srt_content: str) -> str:
    """
    Minimalna konwersja SRT -> ASS.
    Zakłada poprawne timestampy i prosty styl. 
    """
    blocks = srt_content.strip().split('\n\n')
    ass = []
    ass.append("[Script Info]")
    ass.append("Title: Generated Subtitles")
    ass.append("ScriptType: v4.00+")
    ass.append("")
    ass.append("[V4+ Styles]")
    ass.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
               "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
               "Alignment, MarginL, MarginR, MarginV, Encoding")
    # Biały, obrys 2, dół-środek
    ass.append("Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,22,1")
    ass.append("")
    ass.append("[Events]")
    ass.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.splitlines()
        # typowy SRT: idx, time, text...
        if len(lines) >= 3:
            time_line = lines[1]
            text_lines = lines[2:]
            text = "\\N".join(l.strip() for l in text_lines if l.strip())
            if '-->' in time_line:
                start_time, end_time = map(str.strip, time_line.split('-->'))
                start_ass = start_time.replace(',', '.')
                end_ass = end_time.replace(',', '.')
                ass.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
    return "\n".join(ass) + "\n"

# ===========================
# FFmpeg: escapowanie Windows
# ===========================

def escape_path_for_ffmpeg_filter(path: str) -> str:
    """
    Przygotowuje absolutną ścieżkę Windows do użycia w -vf ass=/subtitles=
    Zasady:
      - absolutna ścieżka
      - escapuj dwukropek ':' -> '\:'
      - nie zamieniaj backslashy na slash; ujmujemy całą ścieżkę w pojedyncze cudzysłowy w argumencie filtra
    """
    p = str(Path(path).resolve())
    p = p.replace(":", r"\:")
    return p

def build_vf_ass(ass_path: str) -> str:
    esc = escape_path_for_ffmpeg_filter(ass_path)
    return f"ass='{esc}'"

def build_vf_subtitles(srt_path: str, force_style: str | None = None) -> str:
    esc = escape_path_for_ffmpeg_filter(srt_path)
    if force_style:
        # Wartości stylu bez spacji po przecinkach (łatwiej uniknąć dodatkowego escapowania)
        return f"subtitles='{esc}':force_style='{force_style}'"
    return f"subtitles='{esc}'"

def run_ffmpeg(cmd: list[str]) -> tuple[bool, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout, proc.stderr

def burn_subtitles_with_ffmpeg_windows(input_video: str, srt_content: str, output_video: str, prefer_ass=True) -> tuple[bool, str]:
    """
    - prefer_ass=True: generuje .ass i używa filtra ass; w razie błędu fallback do SRT
    - prefer_ass=False: od razu SRT
    Zwraca (ok, path_or_error)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_abs = str(Path(input_video).resolve())
        out_abs = str(Path(output_video).resolve())

        if prefer_ass:
            # zapis ASS
            ass_path = os.path.join(tmpdir, "subs.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(srt_to_ass(srt_content))

            vf = build_vf_ass(ass_path)
            cmd = [
                "ffmpeg", "-y", "-i", in_abs,
                "-vf", vf,
                "-c:v", "libx264",
                "-c:a", "copy",
                out_abs
            ]
            ok, so, se = run_ffmpeg(cmd)
            if ok and os.path.exists(out_abs):
                return True, out_abs

            # fallback: SRT
            srt_path = os.path.join(tmpdir, "subs.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            vf2 = build_vf_subtitles(srt_path, force_style="FontName=Arial,FontSize=28")
            cmd2 = [
                "ffmpeg", "-y", "-i", in_abs,
                "-vf", vf2,
                "-c:v", "libx264",
                "-c:a", "copy",
                out_abs
            ]
            ok2, so2, se2 = run_ffmpeg(cmd2)
            if ok2 and os.path.exists(out_abs):
                return True, out_abs
            return False, (se or se2)

        else:
            # od razu SRT
            srt_path = os.path.join(tmpdir, "subs.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            vf = build_vf_subtitles(srt_path, force_style="FontName=Arial,FontSize=28")
            cmd = [
                "ffmpeg", "-y", "-i", in_abs,
                "-vf", vf,
                "-c:v", "libx264",
                "-c:a", "copy",
                out_abs
            ]
            ok, so, se = run_ffmpeg(cmd)
            if ok and os.path.exists(out_abs):
                return True, out_abs
            return False, se

# ===========================
# UI
# ===========================

st.set_page_config(page_title="🎬 Transkrypcja i napisy (AI + FFmpeg)", page_icon="🎬", layout="centered")

st.title("🎬 Transkrypcja, SRT i film z wtopionymi napisami")
st.caption("Windows + FFmpeg. Stabilne wtapianie (ASS/SRT) ze wsparciem tłumaczenia.")

languages = {
    "Brak tłumaczenia": None,
    "Polski": "polski",
    "Angielski": "angielski",
    "Niemiecki": "niemiecki",
    "Hiszpański": "hiszpański",
    "Ukraiński": "ukraiński",
}

target_lang_key = st.selectbox("Wybierz język tłumaczenia", list(languages.keys()))
target_lang = languages[target_lang_key]

uploaded_file = st.file_uploader("Wgraj audio/wideo", type=SUPPORTED_FORMATS)

if "processing_results" not in st.session_state:
    st.session_state["processing_results"] = {}

if uploaded_file:
    if uploaded_file.size > MAX_SIZE:
        st.error(f"❌ Plik zbyt duży! Maksymalny rozmiar: {MAX_SIZE // (1024*1024)}MB")
        st.stop()

    st.success(f"✅ Wczytano: {uploaded_file.name} ({uploaded_file.size // (1024*1024)}MB)")
    ext = uploaded_file.name.split('.')[-1].lower()

    # Uwaga: st.video przyjmuje bytes lub ścieżkę; tutaj najpierw pokażemy podgląd po zapisaniu
    if ext in ["mp4", "mov", "avi"]:
        st.info("Podgląd pojawi się po zapisaniu pliku w katalogu tymczasowym.")

    if st.button("🚀 Rozpocznij transkrypcję"):
        client = get_openai_client()
        temp_dir = tempfile.TemporaryDirectory()
        safe_filename = f"input.{ext}"
        input_path = os.path.join(temp_dir.name, safe_filename)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        audio_path = os.path.join(temp_dir.name, "audio.mp3")
        orig_video_path = input_path

        # Podgląd wideo po zapisaniu
        if ext in ["mp4", "mov", "avi"]:
            with open(input_path, "rb") as vf:
                st.video(vf.read())

        # Ekstrakcja audio
        if ext in ["mp4", "mov", "avi"]:
            with st.spinner("🎵 Ekstrakcja audio..."):
                audio = AudioSegment.from_file(input_path)
                audio.export(audio_path, format="mp3")
        else:
            audio_path = input_path

        with st.spinner("🔪 Dzielę audio..."):
            chunk_paths = []
            try:
                chunk_paths = split_audio(audio_path, chunk_length_ms=CHUNK_LENGTH_MS)
            except Exception as e:
                st.error(f"❌ Błąd dzielenia audio: {e}")
                st.stop()

            st.info(f"Ilość fragmentów do transkrypcji: {len(chunk_paths)}")

        full_transcript = ""
        all_segments = []
        for idx, chunk_file in enumerate(chunk_paths):
            with st.spinner(f"🎙️ Transkrypcja fragmentu {idx+1}/{len(chunk_paths)}..."):
                with open(chunk_file, "rb") as f:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json"
                    )
                chunk_result = json.loads(result.model_dump_json())
                offset_sec = (idx * CHUNK_LENGTH_MS) / 1000.0
                for seg in chunk_result.get("segments", []):
                    seg["start"] += offset_sec
                    seg["end"] += offset_sec
                    all_segments.append(seg)
                full_transcript += chunk_result.get("text", "") + " "

        srt_content = generate_srt(all_segments)
        st.text_area("Tekst (transkrypt)", full_transcript.strip(), height=300)
        st.subheader("📺 Napisy SRT")
        with st.expander("Podgląd SRT (pierwsze linie)"):
            preview = srt_content[:800] + ("..." if len(srt_content) > 800 else "")
            st.code(preview)

        st.download_button("📄 Pobierz transkrypcję (TXT)", full_transcript.strip(), file_name="transkrypcja.txt", mime="text/plain")
        st.download_button("📥 Pobierz napisy SRT", srt_content, file_name="napisy.srt", mime="text/plain")

        st.session_state["processing_results"] = {
            "temp_dir": temp_dir,  # UWAGA: obiekt, nie zamykać do końca sesji
            "input_path": input_path,
            "srt_content": srt_content,
            "full_transcript": full_transcript.strip(),
            "all_segments": all_segments,
            "orig_video_path": orig_video_path,
            "ext": ext,
        }

# ===========================
# Generowanie filmu z napisami
# ===========================

pr = st.session_state.get("processing_results", {})
if pr and pr.get("orig_video_path") and pr.get("srt_content") and pr.get("ext") in ["mp4", "mov", "avi"]:
    st.header("🎬 Generowanie filmu z napisami")
    prefer_ass = st.checkbox("Użyj ASS (lepsze style, więcej opcji). Jeśli nie działa, użyjemy fallback SRT.", value=True)

    if st.button("🎬 Wygeneruj film z wtopionymi napisami"):
        with st.spinner("⏳ Tworzenie filmu z napisami..."):
            temp_dir = pr["temp_dir"]  # tymczasowy katalog aktywny
            output_video_path = os.path.join(temp_dir.name, "output_with_subs.mp4")
            ok, result = burn_subtitles_with_ffmpeg_windows(pr["orig_video_path"], pr["srt_content"], output_video_path, prefer_ass=prefer_ass)
            if ok:
                st.success("✅ Film z napisami został utworzony!")
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button("📥 Pobierz film z napisami", data=video_bytes, file_name="film_z_napisami.mp4", mime="video/mp4")
            else:
                st.error("❌ Błąd przy wtapianiu napisów")
                st.code(result)

# ===========================
# Tłumaczenie i generowanie filmu z tłumaczeniem
# ===========================

if pr and pr.get("full_transcript") and target_lang_key != "Brak tłumaczenia":
    if st.button(f"Tłumacz na: {target_lang_key} + pobierz napisy"):
        with st.spinner("Tłumaczenie tekstu..."):
            client = get_openai_client()
            translated_text = translate_long_text(client, pr["full_transcript"], target_lang)
            st.text_area("Tłumaczenie", translated_text, height=300)
            st.download_button(f"📘 Pobierz tłumaczenie ({target_lang_key})", translated_text, file_name=f"tlumaczenie_{target_lang_key}.txt", mime="text/plain")

            # Dopasowanie tłumaczeń do segmentów
            segments = pr["all_segments"]
            translated_lines = [l for l in translated_text.split('\n') if l.strip() != ""]
            translated_srt_lines = []
            for idx, seg in enumerate(segments):
                start = srt_timestamp(seg.get('start', 0))
                end = srt_timestamp(seg.get('end', 0))
                line = translated_lines[idx] if idx < len(translated_lines) else "[Brak tłumaczenia]"
                translated_srt_lines.append(f"{idx+1}\n{start} --> {end}\n{line.strip()}\n")
            translated_srt_content = "\n".join(translated_srt_lines)

            st.subheader("📺 Przetłumaczone SRT")
            with st.expander("Podgląd (pierwsze linie)"):
                st.code(translated_srt_content[:800] + ("..." if len(translated_srt_content) > 800 else ""))

            st.download_button(f"📥 Pobierz SRT ({target_lang_key})", translated_srt_content, file_name=f"napisy_{target_lang_key}.srt", mime="text/plain")

            st.session_state["processing_results"]["translated_srt_content"] = translated_srt_content

if pr and pr.get("ext") in ["mp4", "mov", "avi"] and pr.get("translated_srt_content"):
    if st.button("🎬 Wygeneruj film z tłumaczonymi napisami"):
        with st.spinner("Tworzenie filmu z tłumaczonymi napisami..."):
            output_translated_video_path = os.path.join(pr["temp_dir"].name, "output_with_translated_subs.mp4")
            ok, result = burn_subtitles_with_ffmpeg_windows(pr["orig_video_path"], pr["translated_srt_content"], output_translated_video_path, prefer_ass=True)
            if ok:
                st.success("✅ Film z tłumaczeniem został utworzony!")
                with open(output_translated_video_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button("📥 Pobierz film z tłumaczeniem", data=video_bytes, file_name="film_z_tlumaczeniem.mp4", mime="video/mp4")
            else:
                st.error("❌ Błąd przy wtapianiu tłumaczenia")
                st.code(result)

st.markdown("---")
st.info("ℹ️ Aplikacja używa stabilnego escapowania ścieżek Windows w filtrach FFmpeg (ASS/SRT). "
        "Jeśli napotkasz błędy Invalid argument/original_size, problem zwykle wynika z escapowania ':' i '\\'.")