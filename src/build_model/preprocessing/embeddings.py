from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.config import config

EMBEDDINGS_MODEL = config.embeddings_config.model_to_load
CONTENT_CHUNK_SIZE = config.embeddings_config.content_chunk_size
WORD_TO_TOKEN_RATIO = config.embeddings_config.word_to_token_ratio
ENCODING_BATCH_SIZE = config.embeddings_config.encode_batch_size
EMBEDDINGS_UNKNOWN_MAX_SEQ_LENGTH = 512


def chunk_text_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a text string into overlapping word-based chunks.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def embed_text_with_chunks(
    text: str, model: SentenceTransformer, chunk_size: int, overlap: int
) -> np.ndarray:
    """
    Encode text by splitting into chunks and performing mean pooling of embeddings.
    """
    chunks = chunk_text_by_words(text, chunk_size, overlap)
    if not chunks:
        return np.zeros(
            cast(int, model.get_sentence_embedding_dimension()), dtype=np.float32
        )
    embeddings = model.encode(
        chunks, batch_size=ENCODING_BATCH_SIZE, show_progress_bar=False
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings.mean(axis=0)


def generate_embeddings(
    text_list: list[str],
) -> np.ndarray:
    """
    Generate embeddings
    """
    model = SentenceTransformer(EMBEDDINGS_MODEL, device="cpu")
    max_len = int(
        (model.get_max_seq_length() or EMBEDDINGS_UNKNOWN_MAX_SEQ_LENGTH)
        * (1 / WORD_TO_TOKEN_RATIO)
    )
    chunk_size = int(max_len * CONTENT_CHUNK_SIZE)
    overlap = max_len - chunk_size

    embeddings = [
        embed_text_with_chunks(text, model, chunk_size, overlap)
        for text in tqdm(text_list, desc="Generating embeddings")
    ]

    return np.stack(embeddings)
