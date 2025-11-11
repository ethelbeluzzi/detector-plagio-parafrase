import os
import json
import glob
import joblib
import numpy as np
from scipy import sparse
from typing import List, Dict


def ensure_dir(path: str) -> None:
    """Garante que o diretório existe, criando-o se necessário."""
    os.makedirs(path, exist_ok=True)


def load_corpus(path_raw: str, path_processed: str | None = None) -> List[Dict]:
    """
    Carrega o corpus a partir de data/processed (se existir e não vazio) ou data/raw.
    Retorna uma lista de dicionários:
      { "doc_id": str, "title": str, "text": str, "meta": dict }
    """
    base_path = (
        path_processed
        if path_processed and os.path.exists(path_processed) and os.listdir(path_processed)
        else path_raw
    )
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Diretório não encontrado: {base_path}")

    corpus: List[Dict] = []
    for file_path in sorted(glob.glob(os.path.join(base_path, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        corpus.append(
            {
                "doc_id": doc_id,
                "title": doc_id,
                "text": text,
                "meta": {},
            }
        )
    return corpus


def save_index_lexical(path_out: str, tfidf_model, tfidf_matrix, id_map: List[Dict]) -> None:
    """
    Salva o índice léxico (TF-IDF) e metadados.
    id_map deve ser uma lista de dicts, um por BLOCO:
      {
        "doc_id": str,
        "block_id": int,
        "start_word": int,
        "end_word": int,
        "text": str
      }
    """
    ensure_dir(path_out)
    joblib.dump(tfidf_model, os.path.join(path_out, "tfidf_model.joblib"))
    sparse.save_npz(os.path.join(path_out, "tfidf_matrix.npz"), tfidf_matrix)
    with open(os.path.join(path_out, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def load_index_lexical(path_in: str):
    """
    Carrega o índice léxico (TF-IDF) e metadados (id_map).
    Retorna: (tfidf_model, tfidf_matrix, id_map)
    """
    model = joblib.load(os.path.join(path_in, "tfidf_model.joblib"))
    matrix = sparse.load_npz(os.path.join(path_in, "tfidf_matrix.npz"))
    with open(os.path.join(path_in, "id_map.json"), "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return model, matrix, id_map


def save_index_semantic(path_out: str, embeddings: np.ndarray, id_map: List[Dict], model_name: str) -> None:
    """
    Salva o índice semântico (embeddings) e metadados.
    id_map segue o mesmo formato do índice léxico.
    """
    ensure_dir(path_out)
    np.save(os.path.join(path_out, "embeddings.npy"), embeddings)
    meta = {"id_map": id_map, "model_name": model_name}
    with open(os.path.join(path_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index_semantic(path_in: str):
    """
    Carrega o índice semântico (embeddings) e metadados (id_map + model_name).
    Retorna: (embeddings, id_map, model_name)
    """
    embeddings = np.load(os.path.join(path_in, "embeddings.npy"))
    with open(os.path.join(path_in, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta["id_map"], meta["model_name"]
