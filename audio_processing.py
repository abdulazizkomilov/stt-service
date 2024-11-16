import torch
import logging
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from decouple import config

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = config("MODEL_ID")
print("Using model:", model_id)
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
model.eval()
model.gradient_checkpointing_enable()


def get_asr_result(audio_path, model, processor, sr=16000, chunk_duration=10):
    """Perform ASR on audio in chunks to avoid memory overload."""
    logger.info(f"Performing ASR on {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)(waveform)
    audio = waveform.squeeze().numpy()
    logger.info(f"Audio loaded with {sr} samples per second")
    chunk_samples = sr * chunk_duration
    transcriptions = []

    for start in range(0, len(audio), chunk_samples):
        audio_chunk = audio[start: start + chunk_samples]
        inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        transcriptions.append(transcription)

        del input_values, attention_mask, logits
        torch.cuda.empty_cache()

    logger.info(f"ASR result for {audio_path}: {' '.join(transcriptions)}")

    return " ".join(transcriptions)
