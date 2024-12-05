# RAG (Retrieval-Augmented Generation) System

This project implements a RAG system using LangChain, Hugging Face models, and FAISS for efficient retrieval and question answering.

## Usage

### Clone code branch `improve-emb-n-retrieve` and install all libraries you need.
```
git clone https://github.com/UIT-HuyTanNhiPhuong/Blind-Doctor.git -b improve-emb-n-retrieve
cd Blind-Doctor
```

### Change `config.yaml` file for suitable model.
```
{
    "Speech2Text_PATH": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    "llm_path": "AITeamVN/Vi-Qwen2-3B-RAG",

    "json_documents_path": "rag/informations_vinmec.json",
    "vectorstore_persist_dir": "rag/vinmec_db",
    "embedding_path": "hiieu/halong_embedding",
}
```

### Run app and open vercel online: `https://blind-doctor.vercel.app/` to check this application.
```
uvicorn test_backend:app --reload
```



