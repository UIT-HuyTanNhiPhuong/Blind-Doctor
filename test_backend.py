
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
from gtts import gTTS
import os
from playsound import playsound
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import PyPDF2
import json
from ultils import speech2text, text2speech, get_answer
import base64
# from rag.rag_phase.load_data import load_documents, split_documents
# from rag.rag_phase.create_vectordata import create_embeddings, create_vector_store
# from rag.rag_phase.query_data import create_llm, create_qa_chain
from rag import setup_model_and_tokenizer, create_llm, load_documents_from_json, split_documents, setup_retrievers
app = FastAPI()

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
model_id = 'nguyenvulebinh/wav2vec2-base-vietnamese-250h'
speech2text_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)
speech2text_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

# Set up Model
llm_id = os.getenv('LLM_PATH') # google/gemma-2-9b
model, tokenizer = setup_model_and_tokenizer(llm_id)
llm = create_llm(model, tokenizer)

# Load and process documents
documents = load_documents_from_json('rag/rag-phase/informations_vinmec.json')
texts = split_documents(documents)

# Setup embeddings and retrievers
embedding_id = os.getenv('EMBEDDING_PATH') # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
embeddings = HuggingFaceEmbeddings(model_name=embedding_id)
ensemble_retriever = setup_retrievers(texts, embeddings, 
                                      bm25_k = 1, 
                                      faiss_k = 1, 
                                      faiss_index_path = ,)

# Setup QA Chain
global qa_chain
qa_chain = create_qa_chain(llm, ensemble_retriever)

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
        raise HTTPException(status_code=400, detail="Please provide either audio")

    Question-Answering
    answer, sources = get_answer(question = transcription, 
                        qa_chain = qa_chain)

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
