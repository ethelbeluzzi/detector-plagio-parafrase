import os
from dataclasses import dataclass
from typing import Mapping, Optional

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass


def _to_int(val: Optional[str], default: int) -> int:
    try:
        return int(str(val)) if val is not None else default
    except Exception:
        return default


def _to_float(val: Optional[str], default: float) -> float:
    try:
        return float(str(val)) if val is not None else default
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    # Paths
    DATA_RAW_DIR: str
    DATA_PROCESSED_DIR: str
    DATA_INDEXES_DIR: str
    INDEX_LEX_DIR: str
    INDEX_SEM_DIR: str

    # Retrieval
    K_LEX: int
    K_SEM: int
    K_FINAL: int

    # Combination & thresholds
    ALPHA: float           # peso do score léxico na média ponderada (apenas p/ rank)
    TAU_LEX: float         # limiar BRUTO para plágio literal (léxico)
    TAU_SEM: float         # limiar BRUTO para plágio/paráfrase (semântico)
    DELTA_PARA: float      # separação léxica para classificar paráfrase
    MIN_GATE: float        # filtro anti-ruído em scores brutos

    # Semantic model
    SEM_MODEL_NAME: str

    # Sliding windows (em palavras)
    WINDOW_SIZE: int
    STRIDE: int
    CONTEXT_MARGIN: int


def get_settings(environ: Optional[Mapping[str, str]] = None) -> Settings:
    env = os.environ if environ is None else environ

    data_raw = env.get("DATA_RAW_DIR", "data/raw")
    data_processed = env.get("DATA_PROCESSED_DIR", "data/processed")
    data_indexes = env.get("DATA_INDEXES_DIR", "data/indexes")

    idx_lex = env.get("INDEX_LEX_DIR") or os.path.join(data_indexes, "lexical")
    idx_sem = env.get("INDEX_SEM_DIR") or os.path.join(data_indexes, "semantic")

    # Top-K
    k_lex = _to_int(env.get("K_LEX"), 10)
    k_sem = _to_int(env.get("K_SEM"), 10)
    k_final = _to_int(env.get("K_FINAL"), 5)

    # Combinação & thresholds (BRUTOS)
    alpha = _to_float(env.get("ALPHA"), 0.6)
    tau_lex = _to_float(env.get("TAU_LEX"), 0.85)
    tau_sem = _to_float(env.get("TAU_SEM"), 0.85)
    delta_para = _to_float(env.get("DELTA_PARA"), 0.15)
    min_gate = _to_float(env.get("MIN_GATE"), 0.10)

    # Modelo semântico
    model_name = env.get("SEM_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

    # Janelas
    window_size = _to_int(env.get("WINDOW_SIZE"), 40)
    stride = _to_int(env.get("STRIDE"), 20)
    context_margin = _to_int(env.get("CONTEXT_MARGIN"), 10)

    return Settings(
        DATA_RAW_DIR=data_raw,
        DATA_PROCESSED_DIR=data_processed,
        DATA_INDEXES_DIR=data_indexes,
        INDEX_LEX_DIR=idx_lex,
        INDEX_SEM_DIR=idx_sem,
        K_LEX=k_lex,
        K_SEM=k_sem,
        K_FINAL=k_final,
        ALPHA=alpha,
        TAU_LEX=tau_lex,
        TAU_SEM=tau_sem,
        DELTA_PARA=delta_para,
        MIN_GATE=min_gate,
        SEM_MODEL_NAME=model_name,
        WINDOW_SIZE=window_size,
        STRIDE=stride,
        CONTEXT_MARGIN=context_margin,
    )


settings = get_settings()
