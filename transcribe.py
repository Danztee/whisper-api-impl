import whisperx
import srt
import os
from datetime import timedelta
import sys

device = "cpu"
compute_type = "int8" if device == "cpu" else "float16"

print(f"[INFO] Loading WhisperX model")
model = whisperx.load_model("medium", device, compute_type=compute_type)

def transcribe(file_path, language_code):
    temp_files = []

    print(f"[INFO] Starting transcription for: {file_path} (language: {language_code})")
    file_location = file_path
    temp_files.append(file_location)
    
    srt_file_path = f"{os.path.splitext(file_path)[0]}.srt"
    temp_files.append(srt_file_path)

    process_transcription(file_location, srt_file_path, language_code)
    
    print(f"[INFO] Transcription complete. SRT file at: {srt_file_path}")
    return True


def process_transcription(file_location, srt_file_path, language_code):
    """Process transcription synchronously"""
    print(f"[INFO] Transcribing audio: {file_location}")
        
    result = model.transcribe(file_location)
    align_model, metadata = whisperx.load_align_model(language_code, device=device)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, file_location, device)
    generate_srt(aligned_result["word_segments"], srt_file_path)
    return True

def generate_srt(transcription, file_path="transcription.srt"):
    print(f"[INFO] Composing SRT subtitles...")
    
    subtitles = []
    for index, segment in enumerate(transcription):
        subtitles.append(srt.Subtitle(
            index=index + 1,
            start=timedelta(seconds=segment["start"]),
            end=timedelta(seconds=segment["end"]),
            content=segment["word"]
        ))
    srt_content = srt.compose(subtitles)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return file_path

def remove_temp_files(temp_files):
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass


if __name__ == "__main__":
    print(sys.argv)
    file_path = sys.argv[1]
    language_code = sys.argv[2] or 'en'

    transcribe(file_path, language_code)