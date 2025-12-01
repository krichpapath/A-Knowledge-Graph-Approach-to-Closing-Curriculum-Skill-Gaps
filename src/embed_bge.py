# src/embed_bge.py

from sentence_transformers import SentenceTransformer
from typing import List
from src.config import EMBEDDING_MODEL_NAME

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> List[list]:
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True)
    return vecs.tolist()

