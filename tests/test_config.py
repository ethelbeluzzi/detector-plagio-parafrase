import os
import pytest
from src import config


# 🔹 Testa leitura de variáveis do ambiente com valores customizados
def test_get_settings_custom_env():
    env = {
        "DATA_RAW_DIR": "custom/raw",
        "DATA_PROCESSED_DIR": "custom/processed",
        "DATA_INDEXES_DIR": "custom/indexes",
        "INDEX_LEX_DIR": "custom/lex",
        "INDEX_SEM_DIR": "custom/sem",
        "K_LEX": "15",
        "K_SEM": "12",
        "K_FINAL": "8",
        "ALPHA": "0.75",
        "TAU_LEX": "0.95",
        "TAU_SEM": "0.90",
        "DELTA_PARA": "0.25",
        "MIN_GATE": "0.20",
        "SEM_MODEL_NAME": "fake-model",
        "WINDOW_SIZE": "50",
        "STRIDE": "25",
        "CONTEXT_MARGIN": "12"
    }

    settings = config.get_settings(env)

    assert settings.DATA_RAW_DIR == "custom/raw"
    assert settings.K_LEX == 15
    assert settings.ALPHA == 0.75
    assert settings.SEM_MODEL_NAME == "fake-model"
    assert settings.WINDOW_SIZE == 50
    assert settings.CONTEXT_MARGIN == 12


# 🔹 Testa aplicação de valores padrão quando variáveis não são definidas
def test_get_settings_defaults():
    settings = config.get_settings({})
    assert settings.DATA_RAW_DIR == "data/raw"
    assert settings.K_LEX == 10
    assert settings.ALPHA == 0.6
    assert settings.SEM_MODEL_NAME.startswith("sentence-transformers/")
    assert settings.WINDOW_SIZE == 40

