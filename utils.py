import os
import uuid
import logging
import aiofiles
from fastapi import UploadFile

AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)


def setup_logger():
    """Sets up the logging configuration."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file asynchronously."""
    unique_filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path
