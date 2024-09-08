import unittest
import json
import os
from unittest.mock import patch, MagicMock
from rag import (
    extract_plain_text,
    load_documents_from_json,
    split_documents,
    setup_model_and_tokenizer,
    create_llm,
    normalize_scores,
    setup_retrievers,
    create_qa_chain,
    parse_arguments,
)

class TestRAG(unittest.TestCase):

    def test_extract_plain_text(self):
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": []
        }
        expected_output = "key1: value1\n\nkey2: value2"
        self.assertEqual(extract_plain_text(data), expected_output)

    def test_load_documents_from_json(self):
        test_data = [
            {"title1": {"content": "content1"}},
            {"title2": {"content": "content2"}}
        ]
        with open("test_data.json", "w") as f:
            json.dump(test_data, f)

        documents = load_documents_from_json("test_data.json")
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "content: content1")
        self.assertEqual(documents[0].metadata["title"], "title1")

        os.remove("test_data.json")

    def test_split_documents(self):
        from langchain.docstore.document import Document
        docs = [Document(page_content="a" * 2000)]
        split_docs = split_documents(docs, chunk_size=1000)
        self.assertEqual(len(split_docs), 2)

    @patch("rag.AutoTokenizer")
    @patch("rag.AutoModelForCausalLM")
    def test_setup_model_and_tokenizer(self, mock_model, mock_tokenizer):
        model_name = "test_model"
        setup_model_and_tokenizer(model_name)
        mock_tokenizer.from_pretrained.assert_called_once_with(model_name)
        mock_model.from_pretrained.assert_called_once()

    @patch("rag.pipeline")
    def test_create_llm(self, mock_pipeline):
        model = MagicMock()
        tokenizer = MagicMock()
        create_llm(model, tokenizer)
        mock_pipeline.assert_called_once()

    def test_normalize_scores(self):
        import numpy as np
        scores = np.array([1, 2, 3, 4, 5])
        normalized = normalize_scores(scores)
        self.assertTrue(np.allclose(normalized, [0, 0.25, 0.5, 0.75, 1]))

    @patch("rag.BM25Retriever")
    @patch("rag.FAISS")
    def test_setup_retrievers(self, mock_faiss, mock_bm25):
        texts = [MagicMock()]
        embeddings = MagicMock()
        setup_retrievers(texts, embeddings)
        mock_bm25.from_texts.assert_called_once()
        mock_faiss.load_local.assert_called_once()

    @patch("rag.RetrievalQA")
    def test_create_qa_chain(self, mock_retrieval_qa):
        llm = MagicMock()
        retriever = MagicMock()
        create_qa_chain(llm, retriever)
        mock_retrieval_qa.from_chain_type.assert_called_once()

    def test_parse_arguments(self):
        with patch("sys.argv", ["rag.py", "--json_file", "test.json", "--query", "test query"]):
            args = parse_arguments()
            self.assertEqual(args.json_file, "test.json")
            self.assertEqual(args.query, "test query")
            self.assertEqual(args.model_name, "ricepaper/vi-gemma-2b-RAG")

if __name__ == "__main__":
    unittest.main()
