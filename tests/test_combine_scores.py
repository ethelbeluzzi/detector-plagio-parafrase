import pytest
from src.combine_scores import combine_scores


def test_combine_scores_basic():
    top_lex = [("doc_a", 0.9), ("doc_b", 0.4)]
    top_sem = [("doc_a", 0.8), ("doc_b", 0.95)]

    result = combine_scores(top_lex, top_sem, k_final=2, alpha=0.5)

    # Verifica tamanho e campos básicos
    assert len(result) == 2
    assert all("doc_id" in r for r in result)

    # Garante que os resultados estão ordenados por score_final decrescente
    scores = [r["score_final"] for r in result]
    assert scores == sorted(scores, reverse=True)

    # Garante que todos os doc_ids esperados estão presentes
    doc_ids = [r["doc_id"] for r in result]
    assert set(doc_ids) == {"doc_a", "doc_b"}


def test_combine_scores_empty_inputs():
    assert combine_scores([], [], k_final=3) == []


def test_combine_scores_detect_plagio_literal():
    top_lex = [("doc_a", 0.95)]
    top_sem = [("doc_a", 0.9)]

    result = combine_scores(top_lex, top_sem, k_final=1, tau_lex=0.9, tau_sem=0.85)
    assert "plagio_literal" in result[0]["flags"]
    assert result[0]["match_type"] == "plagio_literal"


def test_combine_scores_detect_parafrase():
    top_lex = [("doc_a", 0.5)]
    top_sem = [("doc_a", 0.9)]

    result = combine_scores(top_lex, top_sem, k_final=1, tau_lex=0.9, tau_sem=0.85, delta_para=0.3)
    assert "parafrase" in result[0]["flags"]
    assert result[0]["match_type"] == "parafrase"


def test_combine_scores_min_gate_filter():
    top_lex = [("doc_a", 0.05)]
    top_sem = [("doc_a", 0.04)]  # ambos abaixo de min_gate padrão (0.1)

    result = combine_scores(top_lex, top_sem, k_final=1)
    assert result[0]["flags"] == []
    assert result[0]["match_type"] is None

