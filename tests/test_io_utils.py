import os
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from src import io_utils


# 🔹 Testa ensure_dir criando diretório inexistente
def test_ensure_dir_creates(tmp_path):
    path = tmp_path / "subdir"
    io_utils.ensure_dir(path)
    assert path.exists() and path.is_dir()


# 🔹 Testa load_corpus carregando de path_raw
def test_load_corpus_from_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file_path = raw_dir / "doc1.txt"
    file_path.write_text("conteúdo de teste", encoding="utf-8")

    corpus = io_utils.load_corpus(str(raw_dir))
    assert len(corpus) == 1
    assert corpus[0]["doc_id"] == "doc1"
    assert corpus[0]["text"] == "conteúdo de teste"


# 🔹 Testa load_corpus usando path_processed se não estiver vazio
def test_load_corpus_prefers_processed(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    (processed_dir / "docX.txt").write_text("texto processado", encoding="utf-8")
    (raw_dir / "docY.txt").write_text("texto cru", encoding="utf-8")

    corpus = io_utils.load_corpus(str(raw_dir), str(processed_dir))
    assert len(corpus) == 1
    assert corpus[0]["doc_id"] == "docX"
    assert "processado" in corpus[0]["text"]


# 🔹 Testa save/load do índice léxico
def test_save_and_load_index_lexical(tmp_path):
    out_dir = tmp_path / "lexical"
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(["teste de tfidf"])
    id_map = [{"doc_id": "doc1", "block_id": 0, "start_word": 0, "end_word": 3, "text": "teste de tfidf"}]

    io_utils.save_index_lexical(str(out_dir), tfidf, matrix, id_map)
    assert (out_dir / "tfidf_model.joblib").exists()
    assert (out_dir / "tfidf_matrix.npz").exists()
    assert (out_dir / "id_map.json").exists()

    model_loaded, matrix_loaded, id_map_loaded = io_utils.load_index_lexical(str(out_dir))
    assert isinstance(model_loaded, TfidfVectorizer)
    assert sparse.issparse(matrix_loaded)
    assert id_map_loaded == id_map


# 🔹 Testa save/load do índice semântico
def test_save_and_load_index_semantic(tmp_path):
    out_dir = tmp_path / "semantic"
    embeddings = np.ones((2, 3))
    id_map = [{"doc_id": "doc1"}, {"doc_id": "doc2"}]
    model_name = "fake-model"

    io_utils.save_index_semantic(str(out_dir), embeddings, id_map, model_name)
    assert (out_dir / "embeddings.npy").exists()
    assert (out_dir / "meta.json").exists()

    emb_loaded, id_map_loaded, model_name_loaded = io_utils.load_index_semantic(str(out_dir))
    assert np.array_equal(embeddings, emb_loaded)
    assert id_map_loaded == id_map
    assert model_name_loaded == model_name
