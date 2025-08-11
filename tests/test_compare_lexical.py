import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from src.compare_lexical import compare_lexical


# 🔹 Fixture para preparar um cenário de teste consistente
# Cria um corpus pequeno, treina um TfidfVectorizer e gera:
# - tfidf_model: o modelo treinado
# - tfidf_matrix: matriz TF-IDF dos documentos
# - id_map: lista de dicionários no formato esperado pela função
@pytest.fixture
def tfidf_setup():
    corpus = [
        "Olá mundo de dados e privacidade",
        "Privacidade de dados pessoais e segurança",
        "Cinema brasileiro e políticas culturais",
    ]
    tfidf_model = TfidfVectorizer()
    tfidf_matrix = tfidf_model.fit_transform(corpus)
    id_map = [{"uid": f"doc_{i}"} for i in range(len(corpus))]
    return corpus, tfidf_model, tfidf_matrix, id_map


# 🔹 Testa se o ranking é retornado corretamente
# Espera que o documento mais similar ao query seja o doc_1,
# e o segundo mais próximo seja o doc_0
def test_compare_lexical_ranking_basic(tfidf_setup):
    _, tfidf_model, tfidf_matrix, id_map = tfidf_setup
    query = "privacidade e dados pessoais"
    result = compare_lexical(query, tfidf_model, tfidf_matrix, id_map, top_n=3)

    assert len(result) == 3
    assert result[0][0] == "doc_1"  # documento mais próximo
    assert result[1][0] == "doc_0"  # segundo mais próximo


# 🔹 Testa comportamento com consulta vazia
# Espera que retorne lista vazia se o texto de entrada for vazio
def test_compare_lexical_empty_query(tfidf_setup):
    _, tfidf_model, tfidf_matrix, id_map = tfidf_setup
    assert compare_lexical("", tfidf_model, tfidf_matrix, id_map) == []


# 🔹 Testa se respeita o limite definido pelo parâmetro top_n
# Aqui top_n=1, então só deve retornar o documento mais similar
def test_compare_lexical_top_n_limit(tfidf_setup):
    _, tfidf_model, tfidf_matrix, id_map = tfidf_setup
    result = compare_lexical("dados pessoais e privacidade", tfidf_model, tfidf_matrix, id_map, top_n=1)
    assert len(result) == 1
