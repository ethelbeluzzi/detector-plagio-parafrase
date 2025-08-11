# src/compare_service.py
# Orquestra a comparação por BLOCOS (janelas) e retorna saída estruturada por bloco.

from typing import List, Dict, Any
from src.config import settings
from src import io_utils
from src import compare_lexical, compare_semantic, combine_scores
from src.preprocess import build_windows, extend_context

# --- Carrega índices apenas uma vez ---
TFIDF_MODEL, TFIDF_MATRIX, ID_MAP_LEX = io_utils.load_index_lexical(settings.INDEX_LEX_DIR)
EMBEDDINGS, ID_MAP_SEM, MODEL_NAME = io_utils.load_index_semantic(settings.INDEX_SEM_DIR)


def _validate_id_map_item(item: Any) -> Dict:
    """Garante que cada entrada de id_map possua metadados mínimos."""
    if isinstance(item, dict):
        required = {"doc_id", "start_word", "end_word", "text"}
        missing = required - set(item.keys())
        if missing:
            return {
                "doc_id": item.get("doc_id", str(item)),
                "block_id": item.get("block_id", -1),
                "start_word": item.get("start_word", -1),
                "end_word": item.get("end_word", -1),
                "text": item.get("text", ""),
            }
        return {
            "doc_id": item["doc_id"],
            "block_id": int(item.get("block_id", -1)),
            "start_word": int(item["start_word"]),
            "end_word": int(item["end_word"]),
            "text": item["text"],
        }
    return {
        "doc_id": str(item),
        "block_id": -1,
        "start_word": -1,
        "end_word": -1,
        "text": "",
    }


def compare(texto_redacao: str) -> List[Dict]:
    """
    Executa a comparação por janelas (blocos) da redação contra os índices léxico e semântico.
    Retorna lista de blocos suspeitos com metadados e scores.
    """
    texto = (texto_redacao or "").strip()
    if not texto:
        return []

    # Janela de análise
    windows = build_windows(
        text=texto,
        window_size=settings.WINDOW_SIZE,
        stride=settings.STRIDE,
    )

    resultados: List[Dict] = []

    for w in windows:
        bloco_text = w["text"]

        # Top-K léxico e semântico
        top_lex = compare_lexical.compare_lexical(
            query_block=bloco_text,
            tfidf_model=TFIDF_MODEL,
            tfidf_matrix=TFIDF_MATRIX,
            id_map=ID_MAP_LEX,
            top_n=settings.K_LEX,
        )

        top_sem = compare_semantic.semantic_top_k(
            query_block=bloco_text,
            embeddings=EMBEDDINGS,
            id_map=ID_MAP_SEM,
            model_name=MODEL_NAME,
            k=settings.K_SEM,
        )

        # Combina e classifica
        combined = combine_scores.combine_scores(
            top_lex=top_lex,
            top_sem=top_sem,
            k_final=settings.K_FINAL,
            alpha=settings.ALPHA,
            tau_lex=settings.TAU_LEX,
            tau_sem=settings.TAU_SEM,
        )

        if not combined:
            continue

        best = combined[0]
        match_type = best.get("match_type")
        if match_type is None:
            continue

        # Busca metadados do bloco
        try:
            idx_match = next(i for i, v in enumerate(ID_MAP_LEX)
                             if (v.get("doc_id") if isinstance(v, dict) else str(v)) == best["doc_id"])
            meta_raw = ID_MAP_LEX[idx_match]
        except StopIteration:
            try:
                idx_match = next(i for i, v in enumerate(ID_MAP_SEM)
                                 if (v.get("doc_id") if isinstance(v, dict) else str(v)) == best["doc_id"])
                meta_raw = ID_MAP_SEM[idx_match]
            except StopIteration:
                meta_raw = best["doc_id"]

        meta = _validate_id_map_item(meta_raw)

        resultados.append({
            "bloco_id": int(w["bloco_id"]),
            "inicio": int(w["start_word"]),
            "fim": int(w["end_word"]),
            "trecho": bloco_text,
            "trecho_contexto": extend_context(
                text=texto,
                start_word=w["start_word"],
                end_word=w["end_word"],
                margin=settings.CONTEXT_MARGIN
            ),
            "tipo": match_type,
            "melhor_candidato": {
                "doc_id": meta["doc_id"],
                "block_id": meta.get("block_id", -1),
                "start_word": meta.get("start_word", -1),
                "end_word": meta.get("end_word", -1),
                "text": meta.get("text", ""),
            },
            "scores": {
                "final": float(best["score_final"]),
                "lex_raw": float(best["score_lex_raw"]),
                "sem_raw": float(best["score_sem_raw"]),
                "lex_norm": float(best["score_lex_norm"]),
                "sem_norm": float(best["score_sem_norm"]),
            }
        })

    resultados.sort(key=lambda r: r["scores"]["final"], reverse=True)
    return resultados
from typing import List, Dict, Any
from src.config import settings
from src import io_utils
from src import compare_lexical, compare_semantic, combine_scores
from src.preprocess import build_windows, extend_context

# --- Carrega índices apenas uma vez ---
TFIDF_MODEL, TFIDF_MATRIX, ID_MAP_LEX = io_utils.load_index_lexical(settings.INDEX_LEX_DIR)
EMBEDDINGS, ID_MAP_SEM, MODEL_NAME = io_utils.load_index_semantic(settings.INDEX_SEM_DIR)


def _validate_id_map_item(item: Any) -> Dict:
    if isinstance(item, dict):
        required = {"doc_id", "start_word", "end_word", "text"}
        missing = required - set(item.keys())
        if missing:
            return {
                "doc_id": item.get("doc_id", str(item)),
                "block_id": item.get("block_id", -1),
                "start_word": item.get("start_word", -1),
                "end_word": item.get("end_word", -1),
                "text": item.get("text", ""),
            }
        return {
            "doc_id": item["doc_id"],
            "block_id": int(item.get("block_id", -1)),
            "start_word": int(item["start_word"]),
            "end_word": int(item["end_word"]),
            "text": item["text"],
        }
    return {
        "doc_id": str(item),
        "block_id": -1,
        "start_word": -1,
        "end_word": -1,
        "text": "",
    }


def compare(texto_redacao: str) -> List[Dict]:
    texto = (texto_redacao or "").strip()
    if not texto:
        return []

    windows = build_windows(
        text=texto,
        window_size=settings.WINDOW_SIZE,
        stride=settings.STRIDE,
    )

    resultados: List[Dict] = []

    for w in windows:
        bloco_text = w["text"]

        top_lex = compare_lexical.compare_lexical(
            query_block=bloco_text,
            tfidf_model=TFIDF_MODEL,
            tfidf_matrix=TFIDF_MATRIX,
            id_map=ID_MAP_LEX,
            top_n=settings.K_LEX,
        )

        top_sem = compare_semantic.semantic_top_k(
            query_block=bloco_text,
            embeddings=EMBEDDINGS,
            id_map=ID_MAP_SEM,
            model_name=MODEL_NAME,
            k=settings.K_SEM,
        )

        combined = combine_scores.combine_scores(
            top_lex=top_lex,
            top_sem=top_sem,
            k_final=settings.K_FINAL,
            alpha=settings.ALPHA,
            tau_lex=settings.TAU_LEX,    # agora BRUTO
            tau_sem=settings.TAU_SEM,    # agora BRUTO
            delta_para=settings.DELTA_PARA,
            min_gate=settings.MIN_GATE,
        )

        if not combined:
            continue

        best = combined[0]
        match_type = best.get("match_type")
        if match_type is None:
            continue

        # Metadados do bloco candidato
        try:
            idx_match = next(i for i, v in enumerate(ID_MAP_LEX)
                             if (v.get("doc_id") if isinstance(v, dict) else str(v)) == best["doc_id"])
            meta_raw = ID_MAP_LEX[idx_match]
        except StopIteration:
            try:
                idx_match = next(i for i, v in enumerate(ID_MAP_SEM)
                                 if (v.get("doc_id") if isinstance(v, dict) else str(v)) == best["doc_id"])
                meta_raw = ID_MAP_SEM[idx_match]
            except StopIteration:
                meta_raw = best["doc_id"]

        meta = _validate_id_map_item(meta_raw)

        resultados.append({
            "bloco_id": int(w["bloco_id"]),
            "inicio": int(w["start_word"]),
            "fim": int(w["end_word"]),
            "trecho": bloco_text,
            "trecho_contexto": extend_context(
                text=texto,
                start_word=w["start_word"],
                end_word=w["end_word"],
                margin=settings.CONTEXT_MARGIN
            ),
            "tipo": match_type,
            "melhor_candidato": {
                "doc_id": meta["doc_id"],
                "block_id": meta.get("block_id", -1),
                "start_word": meta.get("start_word", -1),
                "end_word": meta.get("end_word", -1),
                "text": meta.get("text", ""),
            },
            "scores": {
                "final": float(best["score_final"]),
                "lex_raw": float(best["score_lex_raw"]),
                "sem_raw": float(best["score_sem_raw"]),
                "lex_norm": float(best["score_lex_norm"]),
                "sem_norm": float(best["score_sem_norm"]),
            }
        })

    resultados.sort(key=lambda r: r["scores"]["final"], reverse=True)
    return resultados
