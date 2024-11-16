import logging
from audio_processing import get_asr_result, model, processor
from queue import Queue

logger = logging.getLogger(__name__)


def worker(queue: Queue, transcription_results):
    """Worker function to process audio files from the queue."""
    logger.info(f"Starting worker ...")

    while True:
        task = queue.get()
        if task is None:  # Exit signal
            logger.info("Worker exiting.")
            break

        file_id, audio_path = task
        logger.info(f"Received file {audio_path}")

        try:
            logger.info(f"Processing file {audio_path}")
            transcription = get_asr_result(audio_path, model, processor)
            transcription_results[file_id] = transcription  # Update shared dictionary
            logger.info(f"Transcription for {audio_path}: {transcription}")
        except Exception as e:
            transcription_results[file_id] = "Failed"
            logger.error(f"Error processing file {audio_path}: {e}")
