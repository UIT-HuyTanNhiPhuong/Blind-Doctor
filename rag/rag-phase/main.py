import os
from dotenv import load_dotenv
from load_data import load_documents, split_documents
from create_vectordata import create_embeddings, create_vector_store
from query_data import create_llm, create_qa_chain

def main():
    load_dotenv()

    # Load and process documents
    documents = load_documents('data')
    texts = split_documents(documents)

    # Create embeddings and vector store
    embeddings = create_embeddings(os.getenv('MODEL_PATH'))
    db = create_vector_store(texts, embeddings)

    # Create LLM and QA chain
    llm = create_llm(quantization="4bit")
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = create_qa_chain(llm, retriever)

    # Example usage
    def get_answer(question):
        result = qa_chain({"query": question})
        return result["result"], result["source_documents"]

    question = "Chào?"
    answer, sources = get_answer(question)
    print(f"Câu trả lời: {answer}")
    print("\nNguồn tham khảo:")
    for doc in sources:
        print(f"- {doc.metadata['source']}")

if __name__ == "__main__":
    main()