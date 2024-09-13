from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

from .document import create_splits_from_json, join_docs
from .model import create_llm
from .retriever.base import load_vectorstore_from_documents
from .utils import print_color

def create_chain(config, retriever=None):
    # Load and process documents
    texts = create_splits_from_json(config['json_documents_path'],
                                    chunk_size=256,
                                    chunk_overlap=100)

    # Setup embeddings and retrievers
    embeddings = HuggingFaceEmbeddings(model_name=config['embedding_path'])
    vectorstore = load_vectorstore_from_documents(documents=texts,
                                                  embeddings=embeddings,
                                                  persist_dir=config['vectorstore_persist_dir'])
    retriever = vectorstore.as_retriever()

    ## GENERATE ANSWER
    llm_id = config['llm_path']
    print_color(f"[INFO] Setting up LLMs ({llm_id})...", 'grey')
    llm = create_llm(model_name=llm_id, max_token=300)
    print_color(f"[INFO] Successfully loaded LLMs ({llm_id})", 'green')

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Bạn là một trợ lý chỉ sử dụng Context được trích dẫn dưới đây để trả lời câu hỏi của người dùng. Câu trả lời của bạn gồm khoảng 150 từ và đảm bảo trả lời bằng tiếng Việt.
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
    return rag_chain
