from typing import List, Dict


def _split_words(text: str) -> List[str]:
    """
    Divide por espaços preservando pontuação e capitalização.
    Não normaliza: queremos maximizar a detecção literal no léxico.
    """
    if not text:
        return []
    # Simplíssimo e intencional: manter o texto “como veio”
    # Caso queira tratar múltiplos espaços, o split já lida.
    return text.split()


def build_windows(
    text: str,
    window_size: int,
    stride: int
) -> List[Dict]:
    """
    Constrói janelas deslizantes em unidades de palavras.
    Retorna lista de dicts com:
      - bloco_id: índice sequencial
      - start_word: início (índice de palavra, inclusive)
      - end_word: fim (índice de palavra, exclusivo)
      - text: texto do bloco (reconstruído com espaços)
    """
    words = _split_words(text)
    n = len(words)
    windows: List[Dict] = []

    if n == 0 or window_size <= 0:
        return windows

    bloco_id = 0
    i = 0
    while i < n:
        start = i
        end = min(i + window_size, n)
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            windows.append({
                "bloco_id": bloco_id,
                "start_word": start,
                "end_word": end,
                "text": chunk_text,
            })
            bloco_id += 1
        if end == n:
            break
        i += stride if stride > 0 else window_size

    return windows


def extend_context(
    text: str,
    start_word: int,
    end_word: int,
    margin: int
) -> str:
    """
    Retorna o trecho do texto cobrindo [start_word - margin, end_word + margin],
    limitado aos limites do texto.
    """
    words = _split_words(text)
    n = len(words)
    s = max(0, start_word - margin)
    e = min(n, end_word + margin)
    return " ".join(words[s:e]).strip()
