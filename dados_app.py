"""Carga dos dados do app: população, amostra e metadados, lidos uma vez e guardados em cache.

Os caminhos partem deste arquivo (`Path(__file__)`), e não da pasta de onde o app é
aberto, para que `streamlit run` funcione de qualquer diretório.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

PASTA_DADOS = Path(__file__).resolve().parent / "dados"


@st.cache_data
def carregar() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Devolve (população, amostra, metadados), lidos de `dados/`."""
    pop = pd.read_parquet(PASTA_DADOS / "populacao.parquet")
    amostra = pd.read_parquet(PASTA_DADOS / "amostra.parquet")
    meta = json.loads((PASTA_DADOS / "metadados.json").read_text(encoding="utf-8"))
    return pop, amostra, meta
