
from gtts import gTTS
import os
from playsound import playsound
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import PyPDF2
import io
import base64

def speech2text(speech2text_model, speech2text_tokenizer, audio_input, sample_rate, device = 'cpu'):

  # Resample if necessary
  if sample_rate != 16000:
      resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
      audio_input = resampler(audio_input)

  input_values = speech2text_tokenizer(audio_input.squeeze().numpy(), return_tensors="pt").input_values.to(device)
  with torch.no_grad():
    logits = speech2text_model(input_values)

  predicted_ids = torch.argmax(logits.logits, dim=-1)
  transcription = speech2text_tokenizer.batch_decode(predicted_ids)

  return transcription

def pdf2text(pdf_path):
  new = ""
  with open(pdf_path, "rb") as file:
      reader = PyPDF2.PdfReader(file)
      for page in reader.pages:
          new += page.extract_text() + "\n"

  return new

# def text2speech(text):
#    tts = gTTS(text, tld = 'com.vn', lang = 'vi')
#    tts.save('saved_audio/test.mp3')
#    playsound('saved_audio/test.mp3')

def get_answer(question, qa_chain):
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]

def text2speech(text):
    """Convert text to speech and return the audio data as bytes."""
    tts = gTTS(text, tld='com.vn', lang='vi')
    audio_stream = io.BytesIO()
    tts.save(audio_stream)
    audio_stream.seek(0)  # Reset stream position
    return audio_stream.getvalue()  # Return audio data as bytes
