"""Confere os arquivos de dados gravados por `preparar_dados.py`.

Se `dados/metadados.json` não existir, os testes são pulados.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from calculo import sortear_amostra

PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"


@pytest.fixture(scope="module")
def dados():
    """Metadados, população e amostra gravados; pula se ainda não foram gerados."""
    caminho_meta = PASTA_DADOS / "metadados.json"
    caminho_pop = PASTA_DADOS / "populacao.parquet"
    caminho_amostra = PASTA_DADOS / "amostra.parquet"
    if not (caminho_meta.exists() and caminho_pop.exists() and caminho_amostra.exists()):
        pytest.skip(f"dados ausentes em {PASTA_DADOS}")
    with open(caminho_meta, encoding="utf-8") as arquivo:
        meta = json.load(arquivo)
    return meta, pd.read_parquet(caminho_pop), pd.read_parquet(caminho_amostra)


def test_metadados(dados):
    meta, _, _ = dados
    assert meta["N"] == 6320
    assert meta["n"] == 913
    assert meta["semente"] == 20261112


def test_tamanhos_dos_arquivos(dados):
    _, pop, amostra = dados
    assert len(pop) == 6320
    assert len(amostra) == 913


def test_amostra_dentro_da_populacao_e_sem_repeticao(dados):
    _, pop, amostra = dados
    contratos = amostra["numero_do_contrato"]
    assert contratos.is_unique
    assert set(contratos) <= set(pop["numero_do_contrato"])


def test_sorteio_reproduz_a_amostra_gravada(dados):
    meta, pop, amostra = dados
    de_novo = sortear_amostra(pop, meta["n"], meta["semente"])
    assert list(de_novo["numero_do_contrato"]) == list(amostra["numero_do_contrato"])
