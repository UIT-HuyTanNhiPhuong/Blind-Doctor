import argparse
import os
import json
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np

def extract_plain_text(data):
    plain_text = ""
    for key, value in data.items():
        if isinstance(value, str):
            plain_text += f"{key}: {value}\n\n"
        elif isinstance(value, list) and len(value) == 0:
            continue
    return plain_text.strip()

def load_documents_from_json(file_path):
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

def split_documents(documents, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    texts = text_splitter.split_documents(documents)
    return texts

def setup_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, tokenizer

def create_llm(model, tokenizer):
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1000,
        truncation = True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)
        
def normalize_scores(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))


def setup_retrievers(texts, embeddings, bm25_k, faiss_k, faiss_index_path="faiss_index"):
    bm25_texts = [doc.page_content for doc in texts]
    bm25_retriever = BM25Retriever.from_texts(bm25_texts, metadatas=[{"source": "bm25"}] * len(bm25_texts))
    bm25_retriever.k = 1

    # faiss_index_path = "faiss_index"
    faiss_vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        score_normalizer=normalize_scores
    )
    return ensemble_retriever

def create_qa_chain(llm, retriever):
    template = """Sử dụng thông tin sau đây để trả lời câu hỏi chi tiết. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố tạo ra câu trả lời.

    {context}

    Câu hỏi: {question}
    Trả lời chi tiết:"""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG model parameters")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--model_name", type=str, default="ricepaper/vi-gemma-2b-RAG", help="Name of the model to use")
    parser.add_argument("--query", type=str, required=True, help="Query to process")
    parser.add_argument("--bm25_k", type=int, default=1, help="Number of documents to retrieve with BM25")
    parser.add_argument("--faiss_k", type=int, default=1, help="Number of documents to retrieve with FAISS")
    parser.add_argument("--faiss_index_path", type=str, default="faiss_index", help="Path to save/load FAISS index")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load and process documents
    documents = load_documents_from_json(args.json_file)
    texts = split_documents(documents)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    llm = create_llm(model, tokenizer)

    # Setup embeddings and retrievers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Update setup_retrievers function to use argument values
    def setup_retrievers(texts, embeddings, bm25_k, faiss_k, faiss_index_path):
        bm25_texts = [doc.page_content for doc in texts]
        bm25_retriever = BM25Retriever.from_texts(bm25_texts, metadatas=[{"source": "bm25"}] * len(bm25_texts))
        bm25_retriever.k = bm25_k

        if os.path.exists(faiss_index_path):
            faiss_vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            faiss_vectorstore = FAISS.from_documents(texts, embeddings)
            faiss_vectorstore.save_local(faiss_index_path)

        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": faiss_k})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],
            score_normalizer=normalize_scores
        )
        return ensemble_retriever

    ensemble_retriever = setup_retrievers(texts, embeddings, args.bm25_k, args.faiss_k, args.faiss_index_path)

    # Create QA chain
    qa_chain = create_qa_chain(llm, ensemble_retriever)

    # Process query
    result = qa_chain.invoke({"query": args.query})

    print("CÃ¢u há»i:", args.query)
    print("CÃ¢u tráº£ lá»i:")
    for doc in result['source_documents']:
        print(doc.page_content)
        print('---')

if __name__ == "__main__":
    main()
