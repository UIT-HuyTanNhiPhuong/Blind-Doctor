import os
import yaml
import base64
from dotenv import load_dotenv
import torch
import torchaudio

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

from ultils import speech2text, text2speech
from new_rag.pipeline import create_chain
from new_rag.utils import print_color, post_process_answer

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

## PROJECT SETUP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

if os.path.exists('config.yaml'):
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print_color(f"[INFO] Successfully loaded config.yaml", 'green')

rag_chain = create_chain(config)

# Loading S2T model
global speech2text_model, speech2text_tokenizer
model_id = config['Speech2Text_PATH']
speech2text_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
speech2text_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/question-answering/")
async def question_answering(audio: UploadFile = File(None), text: str = Form(None)):
    """
    Endpoint to generate answer from specific question-context

    Args:
        audio (UploadFile): An Audio Question (.mp3 or .wav)
        text (str): A Text Question

    Returns:
        answer: str
    """
    if audio and text:
        raise HTTPException(status_code=400, detail="Please provide only one input, either audio or text.")

    print(f"Audio: {audio}")
    print(f"Text: {text}")

    if audio:
        # Reading Audio File
        SAVE_DIRECTORY = "saved_audio"
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)

        audio_file_path = os.path.join(SAVE_DIRECTORY, audio.filename)

        with open(audio_file_path, "wb") as buffer:
            buffer.write(await audio.read())

        # Speech-2-Text
        audio_input, sample_rate = torchaudio.load(audio_file_path)
        transcription = speech2text(speech2text_model, speech2text_tokenizer, audio_input, sample_rate, device)
        os.remove(audio_file_path)

        question = transcription[0]
        print(f"Transcription: {question}")
    elif text:
        question = text
    else:
        raise HTTPException(status_code=400, detail="Please provide either audio or text input.")

    # Question-Answering
    answer = rag_chain.invoke(question).split("Answer: ")[-1].strip()

    # Text-2-Speech
    audio_data = text2speech(answer)
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')  # Encode audio data to base64

    response = {
        "answer": answer,
        "audio_data": f"data:audio/mp3;base64,{audio_base64}",
        "audio": audio_data
    }
    if audio:
        response["transcription"] = question

    return response


# Ensure 'saved_audio' directory exists
if not os.path.exists('saved_audio'):
    os.makedirs('saved_audio')

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
