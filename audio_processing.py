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


def get_asr_result(audio_path: str, model, processor, sr: int = 16000, chunk_duration: int = 30) -> str:
    """
    Perform speech-to-text on an audio file, processing it in chunks.
    """
    try:
        # Load audio file as waveform
        logger.info(f"Loading audio file: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=sr)

        # Split the audio into chunks
        chunk_samples = sr * chunk_duration
        num_chunks = len(audio) // chunk_samples + 1

        # Transcribe each chunk and combine the results
        transcriptions = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = (i + 1) * chunk_samples
            audio_chunk = audio[start:end]

            # Process the raw waveform with the processor
            inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt", padding=True)

            # Transfer the input tensor to the same device as the model
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Perform inference with the model
            logger.info(f"Performing inference for chunk {i + 1} of {num_chunks}...")
            with torch.no_grad():
                logits = model(inputs["input_values"], attention_mask=inputs.get("attention_mask")).logits

            # Decode the logits to obtain the transcription
            predicted_ids = torch.argmax(logits, dim=-1)
            chunk_transcription = processor.batch_decode(predicted_ids)[0]
            transcriptions.append(chunk_transcription)

        # Combine all transcriptions
        final_transcription = " ".join(transcriptions)
        logger.info("Transcription complete.")
        return final_transcription

    except Exception as e:
        logger.error(f"Error during ASR processing: {e}")
        raise RuntimeError(f"Error processing audio file {audio_path}: {e}")
