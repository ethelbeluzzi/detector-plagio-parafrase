import functools
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Carregamento lazy + cache do modelo para evitar download/instancia repetida
@functools.lru_cache(maxsize=2)
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # import tardio
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """
    Gera embeddings normalizados (L2) para uma lista de textos.
    """
    model = _get_model(model_name)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb


def semantic_top_k(
    query_block: str,
    embeddings: np.ndarray,
    id_map: List[str],
    model_name: str,
    k: int = 10
) -> List[Tuple[str, float]]:
    """
    Compara UM bloco de texto (query_block) contra embeddings indexados (de blocos).
    Retorna [(block_id_map_entry, score_cosine), ...] ordenado por score desc.
    """
    if not query_block:
        return []

    query_vec = embed_texts([query_block], model_name)  # shape (1, d)
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return [(id_map[i]["uid"], float(scores[i])) for i in top_indices]
