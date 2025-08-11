from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compare_lexical(
    query_block: str,
    tfidf_model: TfidfVectorizer,
    tfidf_matrix,
    id_map: List[str],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Compara UM bloco de texto contra o índice TF-IDF (que deve ter sido criado sobre BLOCOS).
    Retorna [(block_id_map_entry, score_cosine), ...] ordenado por score desc.
    """
    if not query_block:
        return []

    query_vec = tfidf_model.transform([query_block])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_n]
    return [(id_map[i]["uid"], float(scores[i])) for i in top_indices]

