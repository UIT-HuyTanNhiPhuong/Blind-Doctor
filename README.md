# RAG (Retrieval-Augmented Generation) System

This project implements a RAG system using LangChain, Hugging Face models, and FAISS for efficient retrieval and question answering.

## Usage

### Running the RAG script

To run the RAG script, use the following command:
```
python rag.py --json_file <path_to_json_file> --query "<your_query>" [--model_name <model_name>] [--bm25_k <k_value>] [--faiss_k <k_value>] [--faiss_index_path <path>]
```


Arguments:
- `--json_file`: (Required) Path to the JSON file containing the documents.
- `--query`: (Required) The query you want to ask.
- `--model_name`: (Optional) Name of the model to use. Default is "ricepaper/vi-gemma-2b-RAG".
- `--bm25_k`: (Optional) Number of documents to retrieve with BM25. Default is 1.
- `--faiss_k`: (Optional) Number of documents to retrieve with FAISS. Default is 1.
- `--faiss_index_path`: (Optional) Path to save/load FAISS index. Default is "faiss_index".

Example:
```
python rag.py --json_file data.json --query "What is the capital of France?" --model_name "ricepaper/vi-gemma-2b-RAG" --bm25_k 2 --faiss_k 2
```


### Running the tests

To run the test cases, use the following command:
```
python -m unittest test_rag.py
```

This will run all the test cases defined in the `test_rag.py` file.

## File Structure

- `rag.py`: The main script containing the RAG implementation.
- `test_rag.py`: Test cases for the RAG script.
- `README.md`: This file, containing instructions and information about the project.



