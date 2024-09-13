import os
from dotenv import load_dotenv
import torch
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from ultils import print_color
from new_rag.document import create_splits_from_json, join_docs
from new_rag.model import create_llm
from new_rag.retriever.base import load_vectorstore_from_documents

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## PROJECT SETUP
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


## INDEXING
# Load and process documents
texts = create_splits_from_json('rag/informations_vinmec.json',
                                chunk_size=256,
                                chunk_overlap=100)

# Setup embeddings and retrievers
embeddings = HuggingFaceEmbeddings(model_name=os.getenv('EMBEDDING_PATH'))
vectorstore = load_vectorstore_from_documents(documents=texts,
                                              embeddings=embeddings,
                                              persist_dir="rag/vinmec_db")
retriever = vectorstore.as_retriever()

question = "Tại sao lại có sự thay đổi nội tiết tố?"

## GENERATE ANSWER
llm_id = os.getenv('LLM_PATH')
print_color(f"[INFO] Setting up LLMs ({llm_id})...", 'green')
llm = create_llm(model_name=llm_id, max_token=300)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Bạn là một trợ lý chỉ sử dụng ngữ cảnh (Context) được trích dẫn dưới đây để trả lời câu hỏi của người dùng. Câu trả lời của bạn gồm khoảng 150 từ và đảm bảo trả lời bằng tiếng Việt.
Nếu bạn không biết câu trả lời hoặc câu trả lời không có trong Context, hãy nói rằng bạn không biết. Đừng tạo ra câu trả lời mà không sử dụng các nguồn dưới đây. Tuyệt đối không được liệt kê trùng lắp nội dung.

Context: {context}

Question: {question}

Answer: """,
)

rag_chain = (
    {"context": retriever | join_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain and print the answer
answer = rag_chain.invoke(question).split("Answer: ")

print_color("Context:", 'blue')
print(answer[0])

print_color("Answer:", 'blue')
print(answer[1])
