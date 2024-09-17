from gtts import gTTS
import os
from playsound import playsound

def text2speech(text, saved_file = 'saved_audio/test.mp3'):
   tts = gTTS(text, tld = 'com.vn', lang = 'vi')
   tts.save(saved_file)
   playsound(saved_file)

text2speech('Anh ngáo ngơ vì em mà')