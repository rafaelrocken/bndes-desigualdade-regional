"""Carga dos dados do app: população, amostra, metadados, réguas e malhas do IBGE, lidos uma vez e guardados em cache.

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


@st.cache_data
def carregar_simulacao() -> dict:
    """Resultado dos 1.000 sorteios, gravado por `ferramentas/simular_sorteios.py` (o app não simula)."""
    return json.loads((PASTA_DADOS / "simulacao.json").read_text(encoding="utf-8"))


@st.cache_data
def carregar_ibge() -> pd.DataFrame:
    """PIB 2023 e população 2022 por UF (IBGE), gravados por `ferramentas/baixar_ibge.py`."""
    return pd.read_csv(PASTA_DADOS / "ibge_uf.csv", comment="#")


@st.cache_data
def carregar_malhas() -> dict[str, dict]:
    """Malhas do IBGE para o mapa: {"regiao": 5 regiões, "estado": 27 UFs}."""
    return {
        nivel: json.loads((PASTA_DADOS / arquivo).read_text(encoding="utf-8"))
        for nivel, arquivo in (("regiao", "regiao.geojson"), ("estado", "uf.geojson"))
    }
