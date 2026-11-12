"""Fixtures compartilhadas pelos testes do app."""
from pathlib import Path

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
