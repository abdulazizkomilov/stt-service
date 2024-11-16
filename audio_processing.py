import torch
import logging
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from decouple import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load model and processor
model_id = config("MODEL_ID")
logger.info(f"Using model: {model_id}")
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
model.eval()


# Define ASR function
def get_asr_result(audio_path, model, processor, sr=16000, chunk_duration=10):
    """
    Perform ASR on audio in chunks to avoid memory overload.

    Args:
        audio_path (str): Path to the audio file.
        model (Wav2Vec2ForCTC): Pretrained ASR model.
        processor (Wav2Vec2Processor): Processor for handling inputs/outputs.
        sr (int): Target sample rate (default is 16000).
        chunk_duration (int): Chunk duration in seconds to split audio.

    Returns:
        str: Transcribed text.
    """
    logger.info(f"Performing ASR on: {audio_path}")
    try:
        # Load and resample audio
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"Original waveform shape: {waveform.shape}, sample rate: {sample_rate}")

        if waveform.ndim > 1:  # If stereo, convert to mono
            logger.info("Converting stereo to mono")
            waveform = waveform.mean(dim=0)

        if sample_rate != sr:
            logger.info(f"Resampling audio from {sample_rate}Hz to {sr}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
            waveform = resampler(waveform)

        logger.info(f"Resampled waveform shape: {waveform.shape}")
        audio = waveform.squeeze().numpy()

        if len(audio) == 0:
            logger.error("Empty audio file after resampling")
            return "Error: Empty audio"

        chunk_samples = sr * chunk_duration
        transcriptions = []

        # Process audio in chunks
        for start in range(0, len(audio), chunk_samples):
            audio_chunk = audio[start: start + chunk_samples]
            if len(audio_chunk) == 0:
                logger.warning(f"Skipping empty chunk at start index {start}")
                continue

            logger.info(f"Processing chunk from {start} to {start + chunk_samples}")
            try:
                # Prepare input for the model
                inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs["input_values"].to(device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # Perform inference
                with torch.no_grad():
                    with torch.autocast("cuda", enabled=(device == "cuda")):
                        logits = model(input_values, attention_mask=attention_mask).logits

                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcriptions.append(transcription)

            except Exception as chunk_error:
                logger.error(f"Error processing chunk from {start} to {start + chunk_samples}: {chunk_error}")

            # Free memory
            del input_values, attention_mask, logits
            torch.cuda.empty_cache()

        final_transcription = " ".join(transcriptions)
        logger.info(f"ASR result for {audio_path}: {final_transcription}")
        return final_transcription

    except Exception as e:
        logger.error(f"Error during ASR: {e}")
        return "Error during ASR processing"
