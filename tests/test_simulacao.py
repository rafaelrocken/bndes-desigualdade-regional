"""Confere o arquivo da simulação de 1.000 sorteios, `dados/simulacao.json`.

O arquivo é gerado uma vez por `ferramentas/simular_sorteios.py` (o app só o lê).
"""
import json
from pathlib import Path

import pytest

CAMINHO = Path(__file__).resolve().parents[1] / "dados" / "simulacao.json"
HIPOTESES = ["H1", "H2", "H3", "H4", "H5"]
INTERVALOS = ["pi", "spread", "prazo_amortizacao"]


@pytest.fixture(scope="module")
def sim() -> dict:
    assert CAMINHO.exists(), f"rode ferramentas/simular_sorteios.py: falta {CAMINHO}"
    return json.loads(CAMINHO.read_text(encoding="utf-8"))


def test_tem_as_cinco_hipoteses_e_os_tres_intervalos(sim):
    assert sorted(sim["rejeicao"]) == HIPOTESES
    assert sorted(sim["cobertura"]) == sorted(INTERVALOS)


def test_proporcoes_entre_zero_e_um(sim):
    valores = list(sim["rejeicao"].values()) + list(sim["cobertura"].values())
    assert all(0.0 <= v <= 1.0 for v in valores)


def test_mil_sorteios_reprodutiveis(sim):
    assert sim["repeticoes"] == 1000
    assert sim["semente_inicial"] == 1 and sim["semente_final"] == 1000
    assert sim["n"] == 913
    assert len(sim["x_nne"]) == 1000
    assert all(0 <= x <= sim["n"] for x in sim["x_nne"])


def test_coerente_com_a_teoria(sim):
    # cobertura perto de 95% (um pouco acima: os erros padrão ignoram a correção finita)
    assert all(0.93 <= v <= 0.99 for v in sim["cobertura"].values())
    # H2 e H5 têm poder perto de 100%; H1 perto de α
    assert sim["rejeicao"]["H2"] > 0.99 and sim["rejeicao"]["H5"] > 0.95
    assert sim["rejeicao"]["H1"] < 0.15
