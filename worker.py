import logging
from audio_processing import get_asr_result, model, processor
from queue import Queue

logger = logging.getLogger(__name__)


def worker(queue: Queue, transcription_results):
    """Worker function to process audio files from the queue."""
    while True:
        file_id, audio_path = queue.get()
        try:
            if audio_path is None:
                logger.info(f"Worker exiting.")
                break

            logger.info(f"Processing file {audio_path}")
            print("model loaded", model)
            transcription = get_asr_result(audio_path, model, processor)
            transcription_results[file_id] = transcription

            logger.info(f"Transcription for {audio_path}: {transcription}")
        except Exception as e:
            logger.error(f"Error processing file {audio_path}: {e}")
