import os
from dotenv import load_dotenv
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from ultils import print_color
from new_rag.document import load_documents_from_json, split_documents
from new_rag.model import create_llm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## PROJECT SETUP
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## INDEXING
# Setup embeddings and retrievers
persist_dir = "rag/vinmec_db"
embeddings = HuggingFaceEmbeddings(model_name=os.getenv('EMBEDDING_PATH'))

# Load and process documents
documents = load_documents_from_json('rag/informations_vinmec.json')
texts = split_documents(documents)

if os.path.exists(persist_dir):
    print_color(f'[INFO] Load Vectorstore from {persist_dir}...', 'green')
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print_color(f'[INFO] Can\'t find persist_dir ({persist_dir}). Load and Indexing documents...', 'yellow')
    vectorstore = Chroma.from_documents(documents=texts,
                                        embedding=embeddings,
                                        persist_directory=persist_dir)

## RETRIEVAL
bm25_doc_retriever = BM25Retriever.from_documents(texts)
bm25_doc_retriever.k = 3

## GENERATE ANSWER
llm_id = os.getenv('LLM_PATH')
print_color(f"[INFO] Setting up LLMs ({llm_id})...", 'green')
llm = create_llm(model_name=llm_id, max_token=300)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that try to use to context retrieved below to answer user questions. Be brief in your answers and make sure to answer the user's question using Vietnamese.
If you don't know the answer, you can say that you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.

Context: {context}

Question: {question}

Answer: """,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": bm25_doc_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
answer = rag_chain.invoke("Tại sao lại có sự thay đổi nội tiết tố?").split("Answer: ")

print_color("Context:", 'blue')
print(answer[0])

print_color("Answer:", 'blue')
print(answer[1])
