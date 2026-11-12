"""Fixtures compartilhadas pelos testes do app."""
from pathlib import Path

import pandas as pd
import pytest

# app/tests/conftest.py -> parents[0] = tests, [1] = app, [2] = pasta da tarefa
CSV_BRUTO = Path(__file__).resolve().parents[2] / "dados" / "operacoes-nao-automaticas.csv"


@pytest.fixture(scope="session")
def csv_bruto() -> Path:
    """Caminho do CSV bruto do BNDES; pula o teste se ele não existir.

    O repositório público não traz o CSV, então os testes que dependem dele
    são pulados fora da máquina de quem baixou a base.
    """
    if not CSV_BRUTO.exists():
        pytest.skip(f"CSV bruto ausente: {CSV_BRUTO}")
    return CSV_BRUTO


@pytest.fixture
def pop_pequena() -> pd.DataFrame:
    """População sintética de 20 contratos, sem sorteio: todas as colunas, os dois
    grupos, as cinco regiões, os quatro setores e três spreads ausentes.

    Contagem por região (SE 7, NE 5, S 4, CO 3, N 1) sem empates, de modo que a
    ordem decrescente das barras seja única.
    """
    regioes = ["SE"] * 7 + ["NE"] * 5 + ["S"] * 4 + ["CO"] * 3 + ["N"]
    uf_da_regiao = {"N": "PA", "NE": "BA", "CO": "GO", "SE": "SP", "S": "RS"}
    setores = ["INFRAESTRUTURA", "INDUSTRIA", "COMERCIO/SERVICOS", "AGROPECUÁRIA"]
    portes = ["GRANDE", "MÉDIA", "PEQUENA", "MICRO"]
    datas = pd.Timestamp("2010-01-15") + pd.to_timedelta([90 * i for i in range(20)], unit="D")
    spread = [1.5 + 0.17 * i for i in range(20)]
    for i in (3, 11, 17):
        spread[i] = float("nan")
    return pd.DataFrame(
        {
            "numero_do_contrato": [f"1010{i:04d}" for i in range(20)],
            "uf": [uf_da_regiao[r] for r in regioes],
            "regiao": regioes,
            "grupo": ["N+NE" if r in ("N", "NE") else "Demais" for r in regioes],
            "valor_contratado": [1_000_000.0 * (i + 1) ** 1.5 for i in range(20)],
            "spread": spread,
            "prazo_amortizacao": [12.0 * (1 + i % 8) for i in range(20)],
            "porte": [portes[i % 4] for i in range(20)],
            "setor": [setores[i % 4] for i in range(20)],
            "data": datas,
            "ano": datas.year.astype("int32"),
            "cnpj": [f"{i:02d}.000.000/0001-00" for i in range(20)],
            "n_subcreditos": [1 + i % 3 for i in range(20)],
        }
    )


def _encodings(spec: dict):
    """Percorre uma especificação Vega-Lite e devolve cada canal de `encoding`.

    Desce por `layer`, `hconcat`, `vconcat`, `concat` e `spec` (usado por `facet`),
    e inclui também os canais declarados em `facet`. Canais que vêm em lista
    (como `tooltip`) são entregues um a um; só chegam dicionários.
    """
    canais = list(spec.get("encoding", {}).values())
    facetas = spec.get("facet")
    if isinstance(facetas, dict):
        # `facet` é um campo único ou um dicionário {row: ..., column: ...}
        canais += [facetas] if "field" in facetas else list(facetas.values())
    for canal in canais:
        if isinstance(canal, dict):
            yield canal
        elif isinstance(canal, list):
            yield from (c for c in canal if isinstance(c, dict))
    for chave in ("layer", "hconcat", "vconcat", "concat"):
        for sub in spec.get(chave, []):
            yield from _encodings(sub)
    if isinstance(spec.get("spec"), dict):
        yield from _encodings(spec["spec"])
