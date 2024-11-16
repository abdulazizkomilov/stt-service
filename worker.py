import logging
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
            transcription = get_asr_result(audio_path, model, processor)
            transcription_results[file_id] = transcription
        except Exception as e:
            transcription_results[file_id] = "Failed"
            logger.error(f"Error processing file {audio_path}: {e}")
