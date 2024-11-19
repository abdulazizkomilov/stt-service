import uuid
import os
import uvicorn
import librosa
import psutil
import multiprocessing
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from multiprocessing import Process, Queue, Manager
from utils import save_temp_file, setup_logger
from worker import worker

multiprocessing.set_start_method("spawn", force=True)

app = FastAPI()
logger = setup_logger()


@app.get("/")
async def root():
    """Health check."""
    return {"message": "ASR Model is ready for inference"}


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    try:
        file_ids = []
        total_length = 0

        for file in files:
            file_id = str(uuid.uuid4())
            file_path = await save_temp_file(file)
            audio_queue.put((file_id, file_path))
            transcription_results[file_id] = "Processing"
            audio, sr = librosa.load(file_path, sr=None)
            total_length += len(audio) / sr

            file_ids.append(file_id)

        avg_length = total_length / len(files) if files else 0

        return {
            "message": "Audios added to the processing queue",
            "file_ids": file_ids,
            "average_audio_length_seconds": avg_length,
            "estimated_processing_time_minutes": avg_length * len(files) / 60,  # Example formula
        }
    except Exception as e:
        logger.error(f"Failed to add audios to the queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add audios to the queue: {e}")


@app.get("/get_transcription/")
async def get_transcription_result(file_id: str):
    """Endpoint to retrieve transcription result by file ID."""
    try:
        transcription = transcription_results.get(file_id)
        if transcription is None:
            raise HTTPException(status_code=404, detail="Transcription not found or still processing")
        elif transcription == "Processing":
            return {"file_id": file_id, "status": "Processing"}
        return {"file_id": file_id, "transcription": transcription}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error retrieving transcription for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve transcription result")


@app.get("/metrics/")
async def get_system_metrics():
    process = psutil.Process(os.getpid())
    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(),
        "memory_usage_mb": psutil.virtual_memory().used / (1024 ** 2),
        "queue_size": audio_queue.qsize(),
    }
    return metrics


if __name__ == "__main__":
    manager = Manager()
    transcription_results = manager.dict()
    audio_queue = Queue()

    num_workers = 2
    workers = []

    for _ in range(num_workers):
        worker_process = Process(target=worker, args=(audio_queue, transcription_results))
        worker_process.start()
        workers.append(worker_process)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        for _ in workers:
            audio_queue.put(None)
        for worker_process in workers:
            worker_process.join()
