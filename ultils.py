
from collections import Counter
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

def create_answer(question, hits, llm):
  context = ''
  for each in hits:
    context += each['content'] + '\n'
#   template = f"""Bạn là một trợ lý sử dụng Context được trích dẫn dưới đây để trả lời câu hỏi của người dùng. Câu trả lời của bạn gồm khoảng 150 từ và đảm bảo trả lời bằng tiếng Việt.
# Nếu bạn thấy câu trả lời không có trong Context nhưng bạn biết câu trả lời, hãy trả lời câu hỏi theo ý bạn.
# Nếu bạn thấy câu trả lời không có trong Context và bạn cũng không biết câu trả lời, hãy nói rằng bạn không biết. Đừng cố gắng tạo ra 1 câu trả lời mà ngay cả bạn cũng không biết. Tuyệt đối không được liệt kê trùng lắp nội dung

# Context: 
# {context}
# Question: {question}

# Answer: """

#   template = f"""Bạn là một trợ lý trả lời câu hỏi của người dùng. Câu trả lời của bạn gồm khoảng 150 từ và đảm bảo trả lời bằng tiếng Việt.
# Đừng cố gắng tạo ra 1 câu trả lời mà ngay cả bạn cũng không biết. Tuyệt đối không được liệt kê trùng lắp nội dung

# Question: {question}

# Answer: """

  template = f"""Bạn là một trợ lý trả lời câu hỏi của người dùng. Câu trả lời của bạn gồm khoảng 150 từ và đảm bảo trả lời bằng tiếng Việt.
Đừng cố gắng tạo ra 1 câu trả lời mà ngay cả bạn cũng không biết. Tuyệt đối không được liệt kê trùng lắp nội dung
Dưới đây sẽ là những Context để bổ sung cho câu trả lời của bạn. Hãy xem xét Context để xem bạn có thể sử dụng hay không. Nếu có thì bạn sẽ bổ sung thêm câu trả lời bằng Context, còn nếu không thì bạn có thể bỏ qua hoàn toàn Context và trả lời theo ý mình.
Context: 
{context}
Question: {question}
Answer: """

  prompt = template.replace('general: Hỏi', '').replace('Trả lời', '')
  answer = llm.invoke(prompt)

  return answer
def text2speech(text):
    """Convert text to speech and return the audio data as bytes."""
    tts = gTTS(text, tld='com.vn', lang='vi')
    audio_stream = io.BytesIO()
    # tts.save(audio_stream)
    audio_stream.seek(0)  # Reset stream position
    return audio_stream.getvalue()  # Return audio data as bytes

def process_sequence(sequence):
    start_index = sequence.find("Trả lời")
    found_sequence = sequence[start_index:]
    found_sequence = found_sequence.replace("Trả lời", "Trả lời:")

    tokens = found_sequence.split()
    counts = Counter(tokens)

    result = []
    for token in tokens:
        if (counts[token] <= 2 or token not in result):
            result.append(token)

    return ' '.join(result)
