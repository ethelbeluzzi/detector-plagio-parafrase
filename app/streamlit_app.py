import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import base64

# --- Caminho do projeto ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare_service import compare
from src.config import settings


# ========= Helpers de UI =========

def _merge_overlaps(ranges: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Mescla intervalos (start, end, tipo) que se sobrepõem.
    Mantém o tipo "mais severo" quando houver conflito (plágio > paráfrase).
    Severidade: plagio_literal > parafrase.
    """
    if not ranges:
        return []

    severity = {"plagio_literal": 2, "parafrase": 1}
    # Ordena por início
    ranges = sorted(ranges, key=lambda x: (x[0], -severity.get(x[2], 0)))

    merged: List[Tuple[int, int, str]] = []
    cur_s, cur_e, cur_t = ranges[0]

    for s, e, t in ranges[1:]:
        if s <= cur_e:  # overlap
            # estende o fim
            if e > cur_e:
                cur_e = e
            # escolhe tipo mais severo
            if severity.get(t, 0) > severity.get(cur_t, 0):
                cur_t = t
        else:
            merged.append((cur_s, cur_e, cur_t))
            cur_s, cur_e, cur_t = s, e, t

    merged.append((cur_s, cur_e, cur_t))
    return merged


def _highlight_text(query_text: str, resultados: List[Dict]) -> str:
    """
    Destaca no texto original os blocos suspeitos.
    Usa <mark> com classes CSS diferentes para cada tipo.
    """
    words = query_text.split()
    n = len(words)

    # Coleta intervalos suspeitos (em índices de palavra)
    raw_ranges: List[Tuple[int, int, str]] = []
    for r in resultados:
        tipo = r.get("tipo")
        if tipo not in ("plagio_literal", "parafrase"):
            continue
        start_w = max(0, int(r["inicio"]))
        end_w = min(n, int(r["fim"]))
        if end_w > start_w:
            raw_ranges.append((start_w, end_w, tipo))

    if not raw_ranges:
        # Sem destaques: retorna texto simples
        return "<p>" + " ".join(words) + "</p>"

    ranges = _merge_overlaps(raw_ranges)

    # Constrói HTML com marcações
    html_parts = []
    cursor = 0
    for s, e, tipo in ranges:
        # trecho antes
        if s > cursor:
            html_parts.append(" ".join(words[cursor:s]))
        # trecho destacado
        chunk = " ".join(words[s:e])
        if tipo == "plagio_literal":
            html_parts.append(f"<mark class='plagio' title='Plágio literal'>{chunk}</mark>")
        else:
            html_parts.append(f"<mark class='parafrase' title='Paráfrase'>{chunk}</mark>")
        cursor = e
    # restante
    if cursor < n:
        html_parts.append(" ".join(words[cursor:]))

    return "<p>" + " ".join(part for part in html_parts if part) + "</p>"


def _header_from_best(best_block: Dict) -> str:
    """
    Gera o cabeçalho de veredito a partir do melhor bloco.
    """
    tipo = best_block.get("tipo")
    cand = best_block.get("melhor_candidato", {}) or {}
    ref_doc = cand.get("doc_id", "documento do dataset")

    if tipo == "plagio_literal":
        return f"🚨 Esse texto contém <b>trecho com plágio literal</b> de: <b>{ref_doc}</b>"
    if tipo == "parafrase":
        return f"📝 Esse texto contém <b>trecho parafraseado</b> possivelmente derivado de: <b>{ref_doc}</b>"
    return "✅ Não foram encontrados blocos suspeitos com os thresholds atuais."


def _build_top_blocks_df(resultados: List[Dict], limit: int = 5) -> pd.DataFrame:
    rows = []
    for r in resultados[:limit]:
        cand = r.get("melhor_candidato", {}) or {}
        flags = r.get("tipo")
        if flags == "plagio_literal":
            flag_label = "🚨 plágio literal"
        elif flags == "parafrase":
            flag_label = "📝 paráfrase"
        else:
            flag_label = "—"

        rows.append({
            "🔎 Bloco": f"{r['bloco_id']} ({r['inicio']}-{r['fim']})",
            "📄 Documento base": cand.get("doc_id", "—"),
            "🏷 Classificação": flag_label,
            "📊 Score final": f"{r['scores']['final']:.3f}",
            "🅻 Léxico (bruto)": f"{r['scores']['lex_raw']:.3f}",
            "🅻 Léxico (norm.)": f"{r['scores']['lex_norm']:.3f}",
            "🆂 Semântico (bruto)": f"{r['scores']['sem_raw']:.3f}",
            "🆂 Semântico (norm.)": f"{r['scores']['sem_norm']:.3f}",
            "🧩 Trecho candidato": (cand.get("text", "")[:200] + "…") if cand.get("text") and len(cand["text"]) > 200 else (cand.get("text", "") or "—"),
        })
    return pd.DataFrame(rows)


# ========= App =========

def main():
    st.set_page_config(page_title="Case Tecnico - Ethel", layout="wide")

    # Carrega logo em base64
    logo_file = ROOT / "app" / "letrus.png"
    with open(logo_file, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
      /* Cabeçalho */
      .app-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background: white;  /* fundo branco */
        border-bottom: 2px solid #c2c0b8; /* linha separadora */
      }}

      .app-title {{
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: .2px;
        color: #2b2b2b;
      }}

      .app-logo {{
        height: 44px;
        object-fit: contain;
        margin-left: 16px;
      }}

      /* Reduz tamanho do st.title */
      h1 {{
        font-size: 32px !important;
      }}
    </style>

    <div class="app-header">
      <div class="app-title">Case Técnico - Ethel Beluzzi</div>
      <img class="app-logo" src="data:image/png;base64,{logo_base64}" alt="Logo Letrus" />
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Espaço extra entre o traço do header e o título */
    h1 {
        font-size: 32px !important;
        margin-top: 10px !important;  /* ajuste a altura aqui */
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 Comparador de Similaridade de Textos")

    # CSS leve para os destaques
    st.markdown("""
    <style>
      mark.plagio { background-color: #ffb3b3; padding: 0.15rem 0.25rem; border-radius: 4px; }
      mark.parafrase { background-color: #ffe4a3; padding: 0.15rem 0.25rem; border-radius: 4px; }
      .legend span { display: inline-block; margin-right: 16px; }
      .legend .bullet { display:inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }
      .bullet.plagio { background: #ffb3b3; }
      .bullet.parafrase { background: #ffe4a3; }
    </style>
    """, unsafe_allow_html=True)

    query = st.text_area("Cole o texto para análise:", height=220)

    if st.button("Comparar"):
        if not query.strip():
            st.warning("Digite um texto antes de comparar.")
            return

        with st.spinner("Processando..."):
            resultados = compare(query)

        if not resultados:
            st.info("Nenhum bloco suspeito encontrado com os thresholds atuais.")
            return

        # Melhor bloco (ordenado por score_final no serviço)
        best = resultados[0]
        header = _header_from_best(best)
        st.markdown(f"<h2 style='font-size:26px;'>{header}</h2>", unsafe_allow_html=True)

        # Expander com scores e candidatos
        with st.expander("Ver blocos e scores detalhados"):
            # Blocos mais suspeitos (ranking final)
            st.markdown("### 📌 Blocos mais suspeitos")
            df_top = pd.DataFrame([
                {
                    "🔎 Bloco": f"{r['bloco_id']} ({r['inicio']}-{r['fim']})",
                    "📄 Documento base": r["melhor_candidato"].get("doc_id", "—"),
                    "🏷 Classificação": (
                        "🚨 plágio literal" if r["tipo"] == "plagio_literal"
                        else "📝 paráfrase" if r["tipo"] == "parafrase"
                        else "—"
                    ),
                    "📊 Score final": f"{r['scores']['final']:.3f}",
                    "🅻 Léxico (bruto)": f"{r['scores']['lex_raw']:.3f}",
                    "🆂 Semântico (bruto)": f"{r['scores']['sem_raw']:.3f}",
                }
                for r in resultados[:5]
            ])
            st.table(df_top)

            # Mais similares (Léxico) - apenas bruto
            st.markdown("### 📌 Mais similares (Léxico)")
            top_lex = sorted(resultados, key=lambda r: r["scores"]["lex_raw"], reverse=True)[:5]
            df_lex = pd.DataFrame([
                {
                    "🔎 Bloco": f"{r['bloco_id']} ({r['inicio']}-{r['fim']})",
                    "📄 Documento base": r["melhor_candidato"].get("doc_id", "—"),
                    "🅻 Léxico (bruto)": f"{r['scores']['lex_raw']:.3f}",
                }
                for r in top_lex
            ])
            st.table(df_lex)

            # Mais similares (Semântico) - apenas bruto
            st.markdown("### 📌 Mais similares (Semântico)")
            top_sem = sorted(resultados, key=lambda r: r["scores"]["sem_raw"], reverse=True)[:5]
            df_sem = pd.DataFrame([
                {
                    "🔎 Bloco": f"{r['bloco_id']} ({r['inicio']}-{r['fim']})",
                    "📄 Documento base": r["melhor_candidato"].get("doc_id", "—"),
                    "🆂 Semântico (bruto)": f"{r['scores']['sem_raw']:.3f}",
                }
                for r in top_sem
            ])
            st.table(df_sem)

        # Texto com destaques
        st.markdown("### 🖍️ Texto analisado (com destaques)")
        highlighted = _highlight_text(query, resultados)
        st.markdown(highlighted, unsafe_allow_html=True)

        # Legenda de cores
        st.markdown("""
        <div class="legend">
          <span><i class="bullet plagio"></i>Plágio literal</span>
          <span><i class="bullet parafrase"></i>Paráfrase</span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
