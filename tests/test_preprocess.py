import pytest
from src import preprocess


# 🔹 Testa divisão simples de palavras preservando pontuação e capitalização
def test_split_words_basic():
    text = "Olá, Mundo!"
    words = preprocess._split_words(text)
    assert words == ["Olá,", "Mundo!"]

# 🔹 Testa divisão com string vazia
def test_split_words_empty():
    assert preprocess._split_words("") == []


# 🔹 Testa construção de janelas com stride menor que window_size
def test_build_windows_stride_small():
    text = "um dois três quatro cinco seis"
    result = preprocess.build_windows(text, window_size=3, stride=2)
    assert len(result) > 0
    # Primeira janela
    assert result[0]["text"] == "um dois três"
    assert result[0]["start_word"] == 0
    assert result[0]["end_word"] == 3
    # Segunda janela deve começar no índice 2
    assert result[1]["start_word"] == 2


# 🔹 Testa construção de janelas com stride igual ao window_size
def test_build_windows_stride_equals_window():
    text = "a b c d e f"
    result = preprocess.build_windows(text, window_size=2, stride=2)
    assert len(result) == 3  # 6 palavras / 2 por janela
    assert result[0]["text"] == "a b"
    assert result[1]["text"] == "c d"
    assert result[2]["text"] == "e f"


# 🔹 Testa construção de janelas quando window_size é maior que número de palavras
def test_build_windows_window_size_large():
    text = "um dois três"
    result = preprocess.build_windows(text, window_size=10, stride=1)
    assert len(result) == 1
    assert result[0]["text"] == "um dois três"


# 🔹 Testa comportamento com texto vazio ou window_size <= 0
def test_build_windows_edge_cases():
    assert preprocess.build_windows("", window_size=3, stride=1) == []
    assert preprocess.build_windows("texto", window_size=0, stride=1) == []


# 🔹 Testa extensão de contexto com margem dentro dos limites
def test_extend_context_basic():
    text = "um dois três quatro cinco"
    result = preprocess.extend_context(text, start_word=1, end_word=3, margin=1)
    assert result == "um dois três quatro"


# 🔹 Testa extensão de contexto com margem que ultrapassa limites do texto
def test_extend_context_limits():
    text = "um dois três"
    result = preprocess.extend_context(text, start_word=0, end_word=2, margin=5)
    assert result == "um dois três"
