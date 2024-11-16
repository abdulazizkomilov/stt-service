import torch
import logging
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Detect the device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def load_model_and_processor(model_id: str):
    """
    Load the Wav2Vec2 model and processor for speech-to-text.

    Args:
        model_id (str): Path or identifier for the pre-trained model.

    Returns:
        tuple: The model and processor objects.
    """
    logger.info(f"Loading model and processor with ID: {model_id}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
        model.eval()
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model or processor: {e}")
        raise RuntimeError(f"Error loading model or processor: {e}")


def get_asr_result(audio_path: str, model, processor, sr: int = 16000) -> str:
    """
    Perform speech-to-text on an audio file.

    Args:
        audio_path (str): Path to the audio file.
        model: Pretrained Wav2Vec2 model.
        processor: Processor associated with the model.
        sr (int): Target sampling rate for audio processing.

    Returns:
        str: The transcribed text from the audio.
    """
    try:
        # Load audio file as waveform
        logger.info(f"Loading audio file: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=sr)

        # Process the raw waveform with the processor
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        # Transfer the input tensor to the same device as the model
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

        # Perform inference with the model
        logger.info("Performing inference...")
        with torch.no_grad():
            logits = model(inputs["input_values"], attention_mask=inputs.get("attention_mask")).logits

        # Decode the logits to obtain the transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0]

        logger.info("Transcription complete.")
        return predicted_sentence

    except Exception as e:
        logger.error(f"Error during ASR processing: {e}")
        raise RuntimeError(f"Error processing audio file {audio_path}: {e}")
