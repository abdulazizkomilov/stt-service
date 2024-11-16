import torch
import logging
import torchaudio
import librosa
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
def get_asr_result(audio_path, model, processor, sr=16000):
    # Load audio file as a waveform using librosa
    audio, _ = librosa.load(audio_path, sr=sr)

    # Process the raw waveform with the processor
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    # Transfer the input tensor to the same device as the model
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Perform inference with the model
    with torch.no_grad():
        logits = model(inputs["input_values"], attention_mask=inputs.get("attention_mask")).logits

    # Decode the logits to obtain the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentence = processor.batch_decode(predicted_ids)[0]
    return predicted_sentence


result = get_asr_result("./audios/d24e9c76-edd3-4cee-bf17-adc4d264abff.wav", model, processor)
print(result)
