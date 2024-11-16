import uuid
import uvicorn
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
    """Endpoint for receiving audio files and adding them to the processing queue."""
    try:
        file_ids = []
        for file in files:
            file_id = str(uuid.uuid4())
            file_path = await save_temp_file(file)
            audio_queue.put((file_id, file_path))
            transcription_results[file_id] = "Processing"
            file_ids.append(file_id)
        return {"message": "Audios added to the processing queue", "file_ids": file_ids}
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
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    finally:
        for _ in workers:
            audio_queue.put(None)
        for worker_process in workers:
            worker_process.join()
