from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def create_llm(model_id="bigscience/bloom-1b7", max_length=1024, quantization="4bit"):
    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=(quantization == "4bit"),
        load_in_8bit=(quantization == "8bit")
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # Create the HuggingFacePipeline object
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm


def create_qa_chain(llm, retriever):
    template = """Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố tạo ra câu trả lời.

    {context}

    Câu hỏi: {question}
    Trả lời:"""

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