from typing import List, Tuple, Dict, Set


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def combine_scores(
    top_lex: List[Tuple[str, float]],
    top_sem: List[Tuple[str, float]],
    k_final: int,
    alpha: float = 0.6,
    tau_lex: float = 0.85,      # agora: limiar BRUTO (não-normalizado)
    tau_sem: float = 0.85,      # agora: limiar BRUTO (não-normalizado)
    delta_para: float = 0.15,   # separação léxica p/ paráfrase
    min_gate: float = 0.10,     # filtro anti-ruído em scores brutos
) -> List[Dict]:
    """
    Combina resultados léxicos e semânticos para UM bloco.
    - União de candidatos: top_lex ∪ top_sem
    - Normalização apenas para score_final (rank/UX)
    - Classificação (flags/match_type) com scores BRUTOS
    Retorna lista ordenada por score_final desc:
      {
        "doc_id": str, "score_final": float,
        "score_lex_raw": float, "score_sem_raw": float,
        "score_lex_norm": float, "score_sem_norm": float,
        "flags": List[str], "match_type": str | None
      }
    """
    if not top_lex and not top_sem:
        return []

    lex_dict = dict(top_lex)
    sem_dict = dict(top_sem)

    # --- União dos doc_ids preservando diversidade de fontes
    ids: Set[str] = set(lex_dict.keys()) | set(sem_dict.keys())
    if not ids:
        return []

    # Scores BRUTOS alinhados por id
    raw_lex = [float(lex_dict.get(doc_id, 0.0)) for doc_id in ids]
    raw_sem = [float(sem_dict.get(doc_id, 0.0)) for doc_id in ids]
    ids_list = list(ids)

    # Normalização apenas para composição do score_final / exibição
    norm_lex = _normalize_scores(raw_lex)
    norm_sem = _normalize_scores(raw_sem)

    combined: List[Dict] = []
    for doc_id, s_lex_raw, s_sem_raw, s_lex_norm, s_sem_norm in zip(
        ids_list, raw_lex, raw_sem, norm_lex, norm_sem
    ):
        score_final = alpha * s_lex_norm + (1 - alpha) * s_sem_norm

        # --- Classificação com scores BRUTOS ---
        flags: List[str] = []
        match_type = None

        # Filtro anti-ruído: ignora candidatos fracos em ambos os eixos
        if max(s_lex_raw, s_sem_raw) >= min_gate:
            # Plágio literal: alto nos dois eixos
            if (s_lex_raw >= tau_lex) and (s_sem_raw >= tau_sem):
                flags.append("plagio_literal")
                match_type = "plagio_literal"
            # Paráfrase: semântico alto + léxico relativamente baixo
            elif (s_sem_raw >= tau_sem) and (s_lex_raw <= max(tau_lex - delta_para, 0.0)):
                flags.append("parafrase")
                match_type = "parafrase"

        combined.append({
            "doc_id": doc_id,
            "score_final": float(score_final),
            "score_lex_raw": float(s_lex_raw),
            "score_sem_raw": float(s_sem_raw),
            "score_lex_norm": float(s_lex_norm),
            "score_sem_norm": float(s_sem_norm),
            "flags": flags,
            "match_type": match_type,
        })

    combined.sort(key=lambda x: x["score_final"], reverse=True)
    return combined[:k_final]
