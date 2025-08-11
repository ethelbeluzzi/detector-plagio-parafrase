import os
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from src import io_utils
from src.preprocess import build_windows
from src.config import settings


def _encode_embeddings(texts: List[str], model_name: str):
    """
    Carrega modelo SentenceTransformers e gera embeddings normalizados em lote.
    """
    from sentence_transformers import SentenceTransformer  # import tardio para evitar custo no import global
    print(f"[INFO] Carregando modelo de embeddings: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[INFO] Gerando embeddings para {len(texts)} blocos...")
    embeddings = model.encode(
        texts,
        batch_size=32,                 # ajusta se tiver pouca memória
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings


def _flatten_blocks(
    corpus: List[Dict],
    window_size: int,
    stride: int
) -> Tuple[List[str], List[Dict]]:
    """
    Quebra cada documento do corpus em BLOCOS (janelas deslizantes) e
    retorna (lista_textos_blocos, id_map_blocos).
    """
    block_texts: List[str] = []
    id_map: List[Dict] = []

    for doc in corpus:
        doc_id = doc["doc_id"]
        text = doc["text"]
        windows = build_windows(text=text, window_size=window_size, stride=stride)
        for w in windows:
            block_texts.append(w["text"])
            uid = f"{doc_id}#b{w['bloco_id']}"
            id_map.append({
                "uid": uid,
                "doc_id": doc_id,
                "block_id": int(w["bloco_id"]),
                "start_word": int(w["start_word"]),
                "end_word": int(w["end_word"]),
                "text": w["text"],
            })

    return block_texts, id_map


def main():
    path_raw = settings.DATA_RAW_DIR
    path_processed = settings.DATA_PROCESSED_DIR
    path_indexes = settings.DATA_INDEXES_DIR

    print("📂 Carregando corpus...")
    corpus = io_utils.load_corpus(path_raw, path_processed)
    print(f"   • Total de documentos: {len(corpus)}")

    print("🧩 Quebrando documentos em blocos...")
    block_texts, id_map_blocks = _flatten_blocks(
        corpus=corpus,
        window_size=settings.WINDOW_SIZE,
        stride=settings.STRIDE,
    )
    print(f"   • Total de blocos gerados: {len(block_texts)}")

    # ----- Índice Léxico -----
    print("📝 Criando índice léxico (TF-IDF)...")
    tfidf = TfidfVectorizer(
        stop_words=None,
        lowercase=False,
        ngram_range=(1, 5),
    )
    tfidf_matrix = tfidf.fit_transform(block_texts)
    io_utils.save_index_lexical(
        os.path.join(path_indexes, "lexical"),
        tfidf,
        tfidf_matrix,
        id_map_blocks,
    )
    print(f"   • Índice léxico salvo em: {os.path.join(path_indexes, 'lexical')}")

    # ----- Índice Semântico -----
    embeddings = _encode_embeddings(block_texts, model_name=settings.SEM_MODEL_NAME)
    io_utils.save_index_semantic(
        os.path.join(path_indexes, "semantic"),
        embeddings,
        id_map_blocks,
        settings.SEM_MODEL_NAME,
    )
    print(f"   • Índice semântico salvo em: {os.path.join(path_indexes, 'semantic')}")

    print("✅ Índices prontos!")


if __name__ == "__main__":
    main()
