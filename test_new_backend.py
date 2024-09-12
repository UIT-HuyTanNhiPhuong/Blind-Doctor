import os
from dotenv import load_dotenv
import torch
from langchain.retrievers import EnsembleRetriever
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
# Load and process documents
documents = load_documents_from_json('rag/informations_vinmec.json')
texts = split_documents(documents)

# Setup embeddings and retrievers
persist_dir = "rag/vinmec_db"
embeddings = HuggingFaceEmbeddings(model_name=os.getenv('EMBEDDING_PATH'))
if os.path.exists(persist_dir):
    print_color(f'[INFO] Load Vectorstore from {persist_dir}...', 'green')
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print_color(f'[INFO] Can\'t find persist_dir ({persist_dir}). Load and Indexing documents...', 'yellow')
    vectorstore = Chroma.from_documents(documents=texts,
                                        embedding=embeddings,
                                        persist_directory=persist_dir)
retriever = vectorstore.as_retriever()

question = "Tại sao lại có sự thay đổi nội tiết tố?"

bm25_doc_retriever = BM25Retriever.from_documents(texts)
bm25_doc_retriever.k = 3

# Create ensamble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_doc_retriever],
    weights=[0.7, 0.3]
)
docs = ensemble_retriever.invoke(question)

## GENERATE ANSWER
llm_id = os.getenv('LLM_PATH')
print_color(f"[INFO] Setting up LLMs ({llm_id})...", 'green')
llm = create_llm(model_name=llm_id, max_token=300)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Bạn là một trợ lý hữu ích cố gắng sử dụng ngữ cảnh được trích dẫn dưới đây để trả lời câu hỏi của người dùng. Hãy ngắn gọn trong câu trả lời của bạn và đảm bảo trả lời câu hỏi của người dùng bằng tiếng Việt.
Nếu bạn không biết câu trả lời, bạn có thể nói rằng bạn không biết. Đừng tạo ra câu trả lời không sử dụng các nguồn dưới đây và đừng lặp lại từ ngữ.

Context: {context}

Question: {question}

Answer: """,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
answer = rag_chain.invoke(question).split("Answer: ")

print_color("Context:", 'blue')
print(answer[0])

print_color("Answer:", 'blue')
print(answer[1])
