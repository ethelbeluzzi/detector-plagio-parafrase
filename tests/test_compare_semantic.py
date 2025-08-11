import numpy as np
import pytest
from src.compare_semantic import semantic_top_k, embed_texts


# 🔹 Fixture para criar embeddings e id_map fictícios
@pytest.fixture
def semantic_setup():
    # Simula um conjunto de embeddings de 3 documentos
    embeddings = np.array([
        [1.0, 0.0],   # doc_0
        [0.8, 0.2],   # doc_1
        [0.0, 1.0],   # doc_2
    ])
    id_map = [{"uid": f"doc_{i}"} for i in range(len(embeddings))]
    return embeddings, id_map


# 🔹 Testa se o ranking de similaridade está correto
# Usa monkeypatch para evitar carregar modelo real
def test_semantic_top_k_ranking(monkeypatch, semantic_setup):
    embeddings, id_map = semantic_setup

    # Função fake para substituir embed_texts — retorna vetor próximo de doc_1
    def fake_embed_texts(texts, model_name):
        return np.array([[0.8, 0.2]])  # igual a doc_1

    monkeypatch.setattr("src.compare_semantic.embed_texts", fake_embed_texts)

    result = semantic_top_k("texto qualquer", embeddings, id_map, "fake-model", k=3)

    assert len(result) == 3
    assert result[0][0] == "doc_1"  # doc mais próximo
    assert result[1][0] == "doc_0"  # segundo mais próximo


# 🔹 Testa comportamento com query vazia
def test_semantic_top_k_empty_query(semantic_setup):
    embeddings, id_map = semantic_setup
    result = semantic_top_k("", embeddings, id_map, "fake-model")
    assert result == []


# 🔹 Testa se respeita limite k
def test_semantic_top_k_limit(monkeypatch, semantic_setup):
    embeddings, id_map = semantic_setup

    def fake_embed_texts(texts, model_name):
        return np.array([[1.0, 0.0]])  # igual a doc_0

    monkeypatch.setattr("src.compare_semantic.embed_texts", fake_embed_texts)

    result = semantic_top_k("teste", embeddings, id_map, "fake-model", k=1)
    assert len(result) == 1
    assert result[0][0] == "doc_0"

