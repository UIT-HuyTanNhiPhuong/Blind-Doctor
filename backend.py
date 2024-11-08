from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from gtts import gTTS
import os
import base64
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from ultils import speech2text, text2speech, get_answer
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
global qa_chain
embeddings = create_embeddings(os.getenv('MODEL_PATH'))
persist_directory = "rag/rag_phase/chroma_db"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print('Database already exists')
else:
    print('Prepare to create database')
    documents = load_documents('data')
    texts = split_documents(documents)

db = create_vector_store(embeddings=embeddings, persist_directory=persist_directory, texts=texts)

# Create LLM and QA chain
llm = create_llm(model_id="google/gemma-2-9b", quantization="4bit")
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = create_qa_chain(llm, retriever)

# Loading a classification model
classifier = pipeline("text-classification", model="microsoft/deberta-v3-base")

def classify_input(text):
    """
    Xác định domain (y khoa/không y khoa) và loại câu hỏi (trắc nghiệm/tự luận).

    Args:
        text (str): Câu hỏi được chuyển từ audio.

    Returns:
        dict: {'domain': 'y khoa' hoặc 'không y khoa', 'question_type': 'trắc nghiệm' hoặc 'tự luận'}
    """
    medical_keywords = ["bệnh", "triệu chứng", "thuốc", "điều trị", "y học", "khám", "chuẩn đoán"]
    is_medical = any(keyword in text.lower() for keyword in medical_keywords)

    prediction = classifier(text, return_all_scores=True)
    question_type = "trắc nghiệm" if "trắc nghiệm" in text.lower() or max(prediction[0], key=lambda x: x['score'])['label'] == "MULTIPLE_CHOICE" else "tự luận"

    domain = "y khoa" if is_medical else "không y khoa"
    
    return {"domain": domain, "question_type": question_type}

@app.post("/question-answering/")
async def question_answering(file: UploadFile = File(...)):
    """
    Endpoint trả lời câu hỏi từ audio.

    Args:
        file (UploadFile): File âm thanh (.mp3)

    Returns:
        JSON: {'answer': str, 'domain': str, 'question_type': str, 'audio_data': base64}
    """
    # Đọc và lưu file âm thanh
    SAVE_DIRECTORY = "saved_audio"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
  
    audio_file_path = os.path.join(SAVE_DIRECTORY, file.filename)
    with open(audio_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Chuyển đổi Speech-to-Text
    audio_input, sample_rate = torchaudio.load(audio_file_path)
    transcription = speech2text(speech2text_model, speech2text_tokenizer, audio_input, sample_rate, device)
    os.remove(audio_file_path)

    # Phân loại input
    classification = classify_input(transcription)
    domain = classification["domain"]
    question_type = classification["question_type"]

    print(f"Domain: {domain}, Question Type: {question_type}")

    # Kiểm tra domain
    if domain != "y khoa":
        return JSONResponse(content={"error": "Chỉ hỗ trợ câu hỏi liên quan đến y khoa."}, status_code=400)

    # Trả lời câu hỏi
    answer, sources = get_answer(question=transcription, qa_chain=qa_chain)

    # Chuyển Text-to-Speech
    audio_data = text2speech(answer)
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    return {
        "answer": answer,
        "domain": domain,
        "question_type": question_type,
        "audio_data": f"data:audio/mp3;base64,{audio_base64}"
    }

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists('saved_audio'):
    os.makedirs('saved_audio')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
