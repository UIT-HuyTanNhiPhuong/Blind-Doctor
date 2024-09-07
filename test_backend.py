
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from gtts import gTTS
import os
from playsound import playsound
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import PyPDF2
import json
from ultils import speech2text, text2speech, get_answer
import base64
from rag.rag_phase.load_data import load_documents, split_documents
from rag.rag_phase.create_vectordata import create_embeddings, create_vector_store
from rag.rag_phase.query_data import create_llm, create_qa_chain

app = FastAPI()

# Loading S2T model
global speech2text_model, speech2text_tokenizer, device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = os.getenv('Speech2Text_PATH')
speech2text_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
speech2text_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

# Loading RAG-Datbase
# global qa_chain

# embeddings = create_embeddings(os.getenv('MODEL_PATH'))
# persist_directory = "rag/rag-phase/chroma_db"
# if os.path.exists(persist_directory) and os.listdir(persist_directory):
#   print('Database already exists') 
#   text = None
# else : 
#   print(f'Prepare to create database')
#   documents = load_documents('data')
#   texts = split_documents(documents)

# db = create_vector_store(embeddings = embeddings, 
#                          persist_directory = persist_directory, 
#                          texts = texts)

# # Create LLM and QA chain
# llm = create_llm(quantization="4bit")
# retriever = db.as_retriever(search_kwargs={"k": 3})
# qa_chain = create_qa_chain(llm, retriever)

@app.post("/question-answering/")
async def question_answering(file: UploadFile = File(...)):
    """
    Endpoint to generate answer from specific question-context

    Args:
        file (UploadFile): An Audio Question (.mp3)

    Returns:
        answer: str
    """
    # Reading Audio File
    audio_file = f"/tmp/{file.filename}"
    with open(audio_file, "wb") as buffer:
        buffer.write(await file.read())

    # Speech-2-Text
    audio_input, sample_rate = torchaudio.load(audio_file)
    transcription = speech2text(speech2text_model, speech2text_tokenizer, audio_input, sample_rate, device)
    os.remove(audio_file)

    # Question-Answering
    # answer, sources = get_answer(question = transcription, 
    #                     qa_chain = qa_chain)

    answer = 'Làm gì có câu trả lời nào, đang test mà'

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
