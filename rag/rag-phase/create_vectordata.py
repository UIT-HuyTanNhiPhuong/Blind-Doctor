
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

def create_embeddings(model_path):
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def create_vector_store(embeddings, persist_directory="./chroma_db", texts = None):
    # Check if the database already exists
    if texts is None:
        # Load the existing database
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Loaded existing database from:", persist_directory)
    else:
        # Create a new vector store
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        print("Created new database and saved to:", persist_directory)
    return db
