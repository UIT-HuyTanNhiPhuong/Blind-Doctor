
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import PyPDF2

def speech2text(model_id, audio_input, sample_rate, device = 'cpu'):

  tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
  model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

  # Resample if necessary
  if sample_rate != 16000:
      resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
      audio_input = resampler(audio_input)

  input_values = tokenizer(audio_input.squeeze().numpy(), return_tensors="pt").input_values.to(device)
  with torch.no_grad():
    logits = model(input_values)

  predicted_ids = torch.argmax(logits.logits, dim=-1)
  transcription = tokenizer.batch_decode(predicted_ids)

  return transcription

def pdf2text(pdf_path):
  new = ""
  with open(pdf_path, "rb") as file:
      reader = PyPDF2.PdfReader(file)
      for page in reader.pages:
          new += page.extract_text() + "\n"

  return new

# def text2speech(model_id, audio)
