from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
import whisperx
import torch
import srt
import os
import uuid
from datetime import timedelta
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
import asyncio
import uvicorn
from typing import Dict, Optional
import time

# Configure longer timeouts
app = FastAPI()

device = "cpu"
compute_type = "int8" if device == "cpu" else "float16"

# Load WhisperX model
model = whisperx.load_model("medium", device, compute_type=compute_type)

# Load Wav2Vec2 alignment model
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)

# Store transcription jobs
jobs: Dict[str, Dict] = {}

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

def remove_temp_files(temp_files):
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    temp_files = []
    file_location = f"temp_audio_{os.getpid()}.mp3"
    temp_files.append(file_location)
    
    srt_file_path = f"transcription_{os.getpid()}.srt"
    temp_files.append(srt_file_path)

    try:
        # Save uploaded file
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Set timeout for processing
        try:
            # Start transcription in a separate task 
            # with a timeout that's generous enough for processing
            transcription_task = asyncio.create_task(
                asyncio.to_thread(
                    process_transcription, 
                    file_location, 
                    srt_file_path
                )
            )
            
            # Wait for the task to complete with a timeout
            await asyncio.wait_for(transcription_task, timeout=600)  # 10 minutes timeout
            
            # Create a background task to remove temp files after response is sent
            background = BackgroundTask(remove_temp_files, temp_files)
            
            return FileResponse(
                path=srt_file_path,
                filename="transcription.srt",
                media_type="text/plain",
                background=background
            )
            
        except asyncio.TimeoutError:
            remove_temp_files(temp_files)
            raise HTTPException(status_code=504, detail="Processing timed out")

    except Exception as e:
        # Clean up temp files if there's an error
        remove_temp_files(temp_files)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def process_transcription(file_location, srt_file_path):
    """Process transcription synchronously"""
    # Transcribe audio
    result = model.transcribe(file_location)

    # Align transcript for word-level timestamps
    aligned_result = whisperx.align(result["segments"], align_model, metadata, file_location, device)

    # Generate SRT file
    generate_srt(aligned_result["word_segments"], srt_file_path)
    return True

# New job-based API endpoints
@app.post("/job/transcribe/", status_code=status.HTTP_202_ACCEPTED)
async def create_transcription_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Create a new transcription job that runs in the background"""
    job_id = str(uuid.uuid4())
    
    # Create job directory
    os.makedirs("jobs", exist_ok=True)
    job_dir = f"jobs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    
    audio_path = f"{job_dir}/audio.mp3"
    srt_path = f"{job_dir}/transcription.srt"
    
    # Save uploaded file
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job entry
    jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "created_at": time.time(),
        "audio_path": audio_path,
        "srt_path": srt_path,
        "error": None
    }
    
    # Run the transcription in the background
    background_tasks.add_task(
        run_transcription_job,
        job_id,
        audio_path,
        srt_path
    )
    
    return {"job_id": job_id, "status": "processing"}

def run_transcription_job(job_id: str, audio_path: str, srt_path: str):
    """Run transcription job in the background"""
    try:
        # Transcribe audio
        result = model.transcribe(audio_path)
        
        # Align transcript for word-level timestamps
        aligned_result = whisperx.align(result["segments"], align_model, metadata, audio_path, device)
        
        # Generate SRT file
        generate_srt(aligned_result["word_segments"], srt_path)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        # Update job with error
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get the status of a transcription job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "error": job["error"]
    }

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the transcription result if job is completed"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job is still processing")
    
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")
    
    # Create a background task to clean up job files after a day
    def cleanup_job(job_id, delay=86400):
        asyncio.create_task(delayed_cleanup(job_id, delay))
    
    background = BackgroundTask(cleanup_job, job_id)
    
    return FileResponse(
        path=job["srt_path"],
        filename="transcription.srt",
        media_type="text/plain",
        background=background
    )

async def delayed_cleanup(job_id: str, delay: int):
    """Clean up job files after a delay"""
    await asyncio.sleep(delay)
    if job_id in jobs:
        job = jobs[job_id]
        # Remove job files
        remove_temp_files([job["audio_path"], job["srt_path"]])
        # Try to remove job directory
        try:
            os.rmdir(f"jobs/{job_id}")
        except:
            pass
        # Remove job from memory
        del jobs[job_id]

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Remove job files
    remove_temp_files([job["audio_path"], job["srt_path"]])
    
    # Try to remove job directory
    try:
        os.rmdir(f"jobs/{job_id}")
    except:
        pass
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"status": "deleted"}

if __name__ == "__main__":
    # Configure Uvicorn with longer timeouts
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8001, 
        timeout_keep_alive=120,  # Keep-alive timeout
        timeout_graceful_shutdown=120,  # Graceful shutdown timeout
        timeout_notify=30  # When to send notification for timeout
    )

