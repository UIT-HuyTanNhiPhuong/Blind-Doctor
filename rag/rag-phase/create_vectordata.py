from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_embeddings(model_path):
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def create_vector_store(texts, embeddings, persist_directory="./chroma_db"):
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    return db