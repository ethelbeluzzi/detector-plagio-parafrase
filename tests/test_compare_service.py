import pytest
from src import compare_service


def test_compare_service_with_mocked_dependencies(monkeypatch):
    # 🔹 Mock dos índices carregados globalmente
    monkeypatch.setattr(compare_service, "TFIDF_MODEL", "fake-tfidf")
    monkeypatch.setattr(compare_service, "TFIDF_MATRIX", "fake-matrix")
    monkeypatch.setattr(compare_service, "ID_MAP_LEX", [{"doc_id": "doc1", "start_word": 0, "end_word": 3, "text": "abc"}])
    monkeypatch.setattr(compare_service, "EMBEDDINGS", "fake-embeddings")
    monkeypatch.setattr(compare_service, "ID_MAP_SEM", [])
    monkeypatch.setattr(compare_service, "MODEL_NAME", "fake-model")

    # 🔹 Mock de build_windows
    monkeypatch.setattr(compare_service, "build_windows", lambda text, window_size, stride: [
        {"bloco_id": 0, "start_word": 0, "end_word": 3, "text": "teste bloco"}
    ])

    # 🔹 Mock de extend_context
    monkeypatch.setattr(compare_service, "extend_context", lambda text, start_word, end_word, margin: "contexto estendido")

    # 🔹 Mock de compare_lexical
    monkeypatch.setattr(compare_service.compare_lexical, "compare_lexical", lambda **kwargs: [("doc1", 0.9)])

    # 🔹 Mock de compare_semantic
    monkeypatch.setattr(compare_service.compare_semantic, "semantic_top_k", lambda **kwargs: [("doc1", 0.95)])

    # 🔹 Mock de combine_scores
    fake_combined = [{
        "doc_id": "doc1",
        "score_final": 0.95,
        "score_lex_raw": 0.9,
        "score_sem_raw": 0.95,
        "score_lex_norm": 1.0,
        "score_sem_norm": 1.0,
        "flags": ["plagio_literal"],
        "match_type": "plagio_literal"
    }]
    monkeypatch.setattr(compare_service.combine_scores, "combine_scores", lambda **kwargs: fake_combined)

    # 🔹 Executa comparação
    texto = "texto de teste para comparação"
    result = compare_service.compare(texto)

    # 🔹 Validações
    assert isinstance(result, list)
    assert len(result) == 1
    bloco = result[0]
    assert bloco["bloco_id"] == 0
    assert bloco["tipo"] == "plagio_literal"
    assert bloco["melhor_candidato"]["doc_id"] == "doc1"
    assert bloco["scores"]["final"] == 0.95


def test_compare_service_empty_text():
    # 🔹 Texto vazio deve retornar lista vazia
    assert compare_service.compare("") == []
