import os
import numpy as np
import pytest
from src import pipeline_build_index


def test_pipeline_build_index_main(tmp_path, monkeypatch):
    # 🔹 Simula diretórios de entrada/saída no tmp_path
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    indexes_dir = tmp_path / "data" / "indexes"
    for d in [raw_dir, processed_dir, indexes_dir]:
        d.mkdir(parents=True)

    # 🔹 Configura settings falsos
    class FakeSettings:
        DATA_RAW_DIR = str(raw_dir)
        DATA_PROCESSED_DIR = str(processed_dir)
        DATA_INDEXES_DIR = str(indexes_dir)
        WINDOW_SIZE = 5
        STRIDE = 2
        SEM_MODEL_NAME = "fake-model"

    monkeypatch.setattr(pipeline_build_index, "settings", FakeSettings)

    # 🔹 Mock de corpus
    fake_corpus = [{"doc_id": "doc1", "text": "um texto simples para teste"}]
    monkeypatch.setattr(pipeline_build_index.io_utils, "load_corpus", lambda *a, **kw: fake_corpus)

    # 🔹 Mock de build_windows
    monkeypatch.setattr(
        pipeline_build_index, 
        "build_windows", 
        lambda text, window_size, stride: [
            {"bloco_id": 0, "start_word": 0, "end_word": 3, "text": text}
        ]
    )

    # 🔹 Mock de save_index_lexical
    saved_lexical = {}
    def fake_save_lexical(path, tfidf, matrix, id_map):
        saved_lexical["path"] = path
        saved_lexical["n_blocks"] = len(id_map)
    monkeypatch.setattr(pipeline_build_index.io_utils, "save_index_lexical", fake_save_lexical)

    # 🔹 Mock de save_index_semantic
    saved_semantic = {}
    def fake_save_semantic(path, emb, id_map, model_name):
        saved_semantic["path"] = path
        saved_semantic["shape"] = emb.shape
        saved_semantic["model_name"] = model_name
    monkeypatch.setattr(pipeline_build_index.io_utils, "save_index_semantic", fake_save_semantic)

    # 🔹 Mock de _encode_embeddings
    monkeypatch.setattr(pipeline_build_index, "_encode_embeddings", lambda texts, model_name: np.ones((len(texts), 3)))

    # 🔹 Executa pipeline
    pipeline_build_index.main()

    # 🔹 Verificações
    assert saved_lexical["n_blocks"] == 1
    assert saved_semantic["shape"] == (1, 3)
    assert saved_semantic["model_name"] == "fake-model"
    assert os.path.basename(saved_lexical["path"]) == "lexical"
    assert os.path.basename(saved_semantic["path"]) == "semantic"
