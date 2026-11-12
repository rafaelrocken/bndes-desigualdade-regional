"""Testes dos dados do IBGE (réguas por estado e malhas do mapa), gravados por
`ferramentas/baixar_ibge.py`."""
import json
from pathlib import Path

import pandas as pd
import pytest

import calculo as c

PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"


@pytest.fixture(scope="module")
def ibge() -> pd.DataFrame:
    return pd.read_csv(PASTA_DADOS / "ibge_uf.csv", comment="#")


def test_csv_tem_as_27_ufs_e_as_colunas(ibge):
    assert list(ibge.columns) == ["uf", "nome", "regiao", "pib_2023", "populacao_2022"]
    assert len(ibge) == 27 and ibge["uf"].is_unique
    assert set(ibge["regiao"]) == {"N", "NE", "CO", "SE", "S"}
    assert (ibge[["pib_2023", "populacao_2022"]] > 0).all().all()


def test_csv_traz_as_fontes_nos_comentarios():
    cabecalho = (PASTA_DADOS / "ibge_uf.csv").read_text(encoding="utf-8").splitlines()[:4]
    texto = " ".join(cabecalho)
    assert all(linha.startswith("#") for linha in cabecalho)
    assert "apisidra.ibge.gov.br/values/t/5938" in texto and "apisidra.ibge.gov.br/values/t/4709" in texto


def test_norte_e_nordeste_dao_as_reguas_de_h1_e_h2(ibge):
    """As réguas pré-registradas são a soma das partes de N e NE arredondadas a 0,1%."""
    reguas = {h["id"]: h["p0_ou_delta0"] for h in c.HIPOTESES}
    for coluna, regua in (("pib_2023", reguas["H1"]), ("populacao_2022", reguas["H2"])):
        partes = ibge.groupby("regiao")[coluna].sum() / ibge[coluna].sum()
        assert round(round(partes["N"], 3) + round(partes["NE"], 3), 3) == regua
        assert abs(partes["N"] + partes["NE"] - regua) < 0.001  # sem arredondar, a menos de 0,1 ponto


@pytest.mark.parametrize("arquivo, chave, quantas", [("uf.geojson", "uf", 27), ("regiao.geojson", "regiao", 5)])
def test_malhas_tem_as_feicoes_e_cabem_no_app(ibge, arquivo, chave, quantas):
    caminho = PASTA_DADOS / arquivo
    malha = json.loads(caminho.read_text(encoding="utf-8"))
    codigos = [f["properties"][chave] for f in malha["features"]]
    assert len(codigos) == quantas and len(set(codigos)) == quantas
    if chave == "uf":
        assert set(codigos) == set(ibge["uf"])
    assert caminho.stat().st_size < 500_000
