import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


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

def create_llm(model_name=None, model=None, tokenizer=None, max_token = 150):
    # Model_name and model can't be None at the same time
    if model_name is None and model is None:
        raise ValueError("Provide either model_name or model")
    elif model_name is not None:
        model, tokenizer = setup_model_and_tokenizer(model_name)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = max_token,
        truncation = True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)
