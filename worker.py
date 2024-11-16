import logging
from audio_processing import get_asr_result, model, processor
from queue import Queue

logger = logging.getLogger(__name__)


def worker(queue: Queue, transcription_results):
    """Worker function to process audio files from the queue."""
    logger.info(f"Starting worker ...")

    while True:
        task = queue.get()
        if task is None:
            logger.info("Worker exiting.")
            break

        file_id, audio_path = task
        logger.info(f"Received file {audio_path}")

        try:
            logger.info(f"Processing file {audio_path}")
            transcription = get_asr_result(audio_path, model, processor)
            logger.info(f"Finished processing file {audio_path} with transcription: {transcription}")
            transcription_results[file_id] = transcription
            logger.info(f"Transcription for {audio_path}: {transcription}")
        except Exception as e:
            transcription_results[file_id] = "Failed"
            logger.error(f"Error processing file {audio_path}: {e}")
