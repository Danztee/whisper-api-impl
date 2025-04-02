from fastapi import FastAPI, UploadFile, File
import whisperx
import torch
import srt
import os
from datetime import timedelta
from fastapi.responses import FileResponse

app = FastAPI()

device = "cpu"
compute_type = "int8" if device == "cpu" else "float16"

# Load WhisperX model
model = whisperx.load_model("medium", device, compute_type=compute_type)

# Load Wav2Vec2 alignment model
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)

def generate_srt(transcription, file_path="transcription.srt"):
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

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    file_location = "temp_audio.mp3"

    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Transcribe audio
        result = model.transcribe(file_location)

        # Align transcript for word-level timestamps
        aligned_result = whisperx.align(result["segments"], align_model, metadata, file_location, device)

        # Generate SRT file
        srt_file_path = generate_srt(aligned_result["word_segments"])
        
        return FileResponse(srt_file_path, media_type="text/plain", filename="transcription.srt")

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Remove temporary audio file
        if os.path.exists(file_location):
            os.remove(file_location)

