
from rag import load_documents_from_json, split_documents
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers import util
import torch
import os

documents = load_documents_from_json('rag/informations_vinmec.json')
texts = split_documents(documents, chunk_size = 500)
global contents
contents = [each.page_content for each in texts]

model_name = 'vinai/phobert-base-v2' # Name of Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Bi-Encoder
global bi_encoder
bi_encoder = SentenceTransformer(model_name).to(device)
bi_encoder.max_seq_length = 500

def bi_encoder_fn(query, corpus_file = 'bi_cross_encoder/corpus_embeddings.pth', top_k = 25):

  # Load Corpus Embdedings
  if os.path.exists(corpus_file): corpus_embeddings = torch.load(corpus_file, map_location = torch.device(device))
  else : corpus_embeddings = bi_encoder.encode(contents, convert_to_tensor=True, show_progress_bar=True).to(device)

  with torch.no_grad():
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True).to(device) # Embeddings
  hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0] # Calculate Cosine

  for hit in hits:
    corpus_id = hit['corpus_id']
    hit['content'] = contents[corpus_id]

  return hits

# Cross-Encoder
global cross_encoder
cross_encoder = CrossEncoder(model_name, device=device)

def cross_encoder_fn(query, hits, top_k = 3):

  cross_inp = [[query, contents[hit['corpus_id']]] for hit in hits]
  with torch.no_grad():
    cross_scores = cross_encoder.predict(cross_inp)

  # Add Cross-Score
  for idx in range(len(cross_scores)):
    hits[idx]['cross-score'] = cross_scores[idx]

  new_hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
  return new_hits[:top_k]
