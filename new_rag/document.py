import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_plain_text(data):
    """ Extract plain text from data """
    plain_text = ""
    for key, value in data.items():
        if isinstance(value, str):
            plain_text += f"{key}: {value}\n\n"
        elif isinstance(value, list) and len(value) == 0:
            continue
    return plain_text.strip()

def load_documents_from_json(file_path):
    """ Load documents from json file """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    documents = []
    for idx in range(len(data)):
        title = list(data[idx].keys())[0]
        value = list(data[idx].values())[0]
        content = extract_plain_text(value)
        doc = Document(page_content=content, metadata={"title": title})
        documents.append(doc)
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """ Split documents into chunks """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

def create_splits_from_json(file_path, chunk_size=1000, chunk_overlap=0):
    """ Create splits from json file """
    documents = load_documents_from_json(file_path)
    return split_documents(documents, chunk_size, chunk_overlap)

def join_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

