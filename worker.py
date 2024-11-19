import os
import time
import logging
import psutil
from audio_processing import get_asr_result, load_model_and_processor
from multiprocessing import Queue
from decouple import config

model_id = config("MODEL_ID")

logger = logging.getLogger(__name__)


def worker(queue: Queue, transcription_results):
    """Worker function to process audio files from the queue."""
    model, processor = load_model_and_processor(model_id)

    while True:
        task = queue.get()
        if task is None:
            break

        file_id, audio_path = task
        try:
            start_time = time.time()
            transcription = get_asr_result(audio_path, model, processor)

            inference_time = time.time() - start_time
            transcription_results[file_id] = {
                "transcription": transcription,
                "inference_time": inference_time,
            }

            process = psutil.Process(os.getpid())
            cpu_usage = process.cpu_percent(interval=0.1)
            memory_usage = process.memory_info().rss / (1024 ** 2)
            logger.info(f"File {file_id} - Inference Time: {inference_time:.2f}s, "
                        f"CPU: {cpu_usage}%, Memory: {memory_usage:.2f}MB")

            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Successfully deleted processed audio file: {audio_path}")


        except Exception as e:
            transcription_results[file_id] = {"transcription": "Failed", "error": str(e)}
            logger.error(f"Error processing file {audio_path}: {e}")
