import os

from langchain_community.vectorstores import Chroma

from ..utils import print_color

def load_vectorstore_from_documents(documents, embeddings, persist_dir):
    if os.path.exists(persist_dir):
        print_color(f'[INFO] Load Vectorstore from {persist_dir}...', 'green')
        vectorstore = Chroma(persist_directory=persist_dir,
                             embedding_function=embeddings)
    else:
        print_color(f'[INFO] Can\'t find persist_dir ({persist_dir}). Load and Indexing documents...', 'yellow')
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            persist_directory=persist_dir)
    return vectorstore