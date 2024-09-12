from argparse import Namespace

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings


class EmbeddingModel(nn.Module):
    def __init__(self, model_args):
        super(EmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.model = AutoModel.from_pretrained(model_args.model_name_or_path)
        self.max_length = model_args.max_length
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        with torch.no_grad():  # Ensure no gradients are computed for this step
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.transpose(1, 2)
        pooled_output = self.avg_pooling(embeddings)  # Shape: [total_sentences, hidden_size, 1]
        pooled_output = pooled_output.squeeze(dim=-1)  # Shape: [total_sentences, hidden_size]
        return pooled_output  # [n, 768]

class ParagraphTransformer(nn.Module):
    def __init__(self, model_args):
        super(ParagraphTransformer, self).__init__()
        self.device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = model_args.batch_size
        self.embedding = EmbeddingModel(model_args).to(self.device1)

        self.linear = nn.Linear(model_args.dimension, 768).to(self.device1)
        self.LayerNorm = nn.LayerNorm(768).to(self.device1)
        self.dropout = nn.Dropout(p=0.1).to(self.device1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1).to(self.device1)

    def forward(self, features_list):
        embeddings = []

        # Iterate over the list in batches
        for i in tqdm(range(0, len(features_list), self.batch_size), desc="Embedding "):
            batch = features_list[i:i + self.batch_size]  # Ensure `features_list` is used, not `para_post`

            # Assuming `embedding` is a model or callable that processes the batch
            batch_embeddings = self.new_forward(batch)  # Call the model's forward method on the batch

            # Extend the embeddings list with the processed batch
            embeddings.extend(batch_embeddings)  # Append the new batch embeddings

        # Convert the list of tensors to a single tensor
        stacked_tensor = torch.stack(embeddings, dim=0)

        # Move the tensor to the specified device
        return stacked_tensor.to(self.device1)

    def new_forward(self, features_list):  # input is
        # Flatten the list of lists into a single list of sentences
        flattened_features = [sentence for sublist in features_list for sentence in sublist]

        # Get embeddings for all sentences at once
        embeddings = self.embedding(flattened_features)  # [total_sentences, 768]
        embeddings = embeddings.to(self.device1)
        embeddings.requires_grad_()

        # Apply linear transformation, normalization, and dropout
        out = self.linear(embeddings)  # Shape: [total_sentences, 768]
        out = self.LayerNorm(out)  # Shape: [total_sentences, 768]
        out = self.dropout(out)  # Shape: [total_sentences, 768]

        # Now reshape and reduce to [n, 768] by averaging each sublist
        sentence_counts = [len(sublist) for sublist in features_list]  # Number of sentences in each sublist
        reshaped_out = torch.split(out, sentence_counts)  # Split based on the sentence counts
        final_output = torch.stack([sublist.mean(dim=0) for sublist in reshaped_out])  # Shape: [n, 768] -> [ n, 768]

        return final_output.to(self.device1)

class ParagraphTransformerEmbeddings(EmbeddingFunction):
    def __init__(self, model_args=None):
        default_model_args = {
            'model_name_or_path': 'vinai/phobert-base',
            'batch_size': 16,
            'max_length': 256,
            'dimension': 768
        }
        default_model_args = Namespace(**default_model_args)
        self.model = ParagraphTransformer(model_args=default_model_args)

    def embed_query(self, query: str) -> list:
        embedding = self.model([query])
        print(embedding.shape)
        return embedding[0].tolist()

    def embed_documents(self, documents: Documents) -> Embeddings:
        embeddings = self.model(documents)
        return embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_documents(input)

def create_paragraph_transformer():
    model_args = {
        'model_name_or_path': 'vinai/phobert-base',
        'batch_size': 16,
        'max_length': 256,
        'dimension': 768
    }
    model_args = Namespace(**model_args)
    return ParagraphTransformer(model_args=model_args)