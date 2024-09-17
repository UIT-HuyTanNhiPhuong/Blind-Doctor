from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from bi_cross.encoder import bi_encoder_fn, cross_encoder_fn
from huggingface_hub import login
from rag import setup_model_and_tokenizer, create_llm
import os
import torch
import torchaudio
import base64
from ultils import speech2text, text2speech, process_sequence, create_answer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

app = FastAPI()
os.environ["HF_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
login(token=os.environ["HF_API_TOKEN"])

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading S2T model
global speech2text_model, speech2text_tokenizer, device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = os.getenv('Speech2Text_PATH')
speech2text_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
speech2text_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

# Set up Model
llm_id = os.getenv('LLM_PATH')
model, tokenizer = setup_model_and_tokenizer(llm_id)
global llm
llm = create_llm(model, tokenizer)

@app.post("/question-answering/")
async def question_answering(audio: UploadFile = File(None), text: str = None):
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

        question = transcription
    elif text:
        question = text
    else:
        # raise HTTPException(status_code=400, detail="Please provide either audio or text input.")
        question = 'Bác sĩ ơi, tôi bị ốm, tôi phải làm gì?' # Temporary question.

    # Question-Answering
    top_k_bi = 25
    top_k_cross = 3
    corpus_file = 'bi_cross/corpus_embeddings.pth'
    # Bi-Cross-Encoder
    bi_hits = bi_encoder_fn(question, corpus_file = corpus_file, top_k = top_k_bi)
    cross_hits = cross_encoder_fn(question, bi_hits, top_k = top_k_cross)

    answer, sources = create_answer(question = question, 
                                    hits = cross_hits,
                                    llm = llm)

    # Text-2-Speech
    audio_data = text2speech(answer)
    audio_base64 = base64.b64encode(audio_data).decode('utf-8') # Encode audio data to base64

    return {
        "answer": answer,
        "audio_data": f"data:audio/mp3;base64,{audio_base64}"
    }

# Ensure 'saved_audio' directory exists
if not os.path.exists('saved_audio'):
    os.makedirs('saved_audio')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
