"""Prepara os dados do trabalho final da IA003 e sorteia a amostra, uma única vez.

Lê o CSV bruto das operações não automáticas do BNDES, monta a população de
contratos, calcula o tamanho da amostra, sorteia a amostra aleatória simples com a
semente pré-registrada e grava três arquivos em `dados/`:

- `populacao.parquet`: os 6.320 contratos da população;
- `amostra.parquet`: os 913 contratos sorteados;
- `metadados.json`: fonte, tamanhos, semente e o commit do pré-registro.

O app e o notebook só leem esses arquivos; não sorteiam de novo.

Uso: python -X utf8 preparar_dados.py <caminho do CSV>
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from calculo import montar_populacao, sortear_amostra, tamanho_amostra

PASTA_DADOS = Path(__file__).parent / "dados"

# Valores pré-registrados (hipóteses H1 a H5 e semente gravadas antes do sorteio).
SEMENTE = 20261112
COMMIT_PRE_REGISTRO = "c47f9033554cf2e07c9cac692754291c413c1d55"

# Conferência da base bruta baixada em 2026-11-12.
LINHAS_BASE_ESPERADAS = 24035
COLUNAS_BASE_ESPERADAS = 34

FONTE = {
    "fonte_nome": (
        "BNDES — Operações de Financiamento: operações não automáticas "
        "(Portal de Dados Abertos do BNDES)"
    ),
    "fonte_url": "https://dadosabertos.bndes.gov.br/dataset/operacoes-financiamento",
    "arquivo_url": (
        "https://dadosabertos.bndes.gov.br/dataset/10e21ad1-568e-45e5-a8af-43f2c05ef1a2"
        "/resource/6f56b78c-510f-44b6-8274-78a5b7e931f4/download/"
        "operacoes-financiamento-operacoes-nao-automaticas.csv"
    ),
    "atualizado_no_portal": "2026-11-11",
    "baixado_em": "2026-11-12",
}


def dimensoes_csv_bruto(caminho_csv: Path) -> tuple[int, int]:
    """Linhas e colunas do CSV como ele chega, antes de filtrar e de acrescentar `ano`."""
    bruto = pd.read_csv(caminho_csv, sep=";", encoding="latin-1", dtype=str)
    return bruto.shape


def preparar(caminho_csv: Path, pasta_saida: Path = PASTA_DADOS) -> dict:
    """Monta a população, sorteia a amostra e grava os três arquivos. Devolve os metadados."""
    linhas_base, colunas_base = dimensoes_csv_bruto(caminho_csv)
    assert (linhas_base, colunas_base) == (LINHAS_BASE_ESPERADAS, COLUNAS_BASE_ESPERADAS), (
        f"base bruta com {linhas_base} linhas e {colunas_base} colunas; "
        f"esperado {LINHAS_BASE_ESPERADAS} e {COLUNAS_BASE_ESPERADAS}"
    )

    pop = montar_populacao(caminho_csv)
    N = len(pop)
    tam = tamanho_amostra(N)
    n = tam["n"]
    amostra = sortear_amostra(pop, n=n, semente=SEMENTE)

    pasta_saida.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(pasta_saida / "populacao.parquet", index=False)
    amostra.to_parquet(pasta_saida / "amostra.parquet", index=False)

    metadados = {
        **FONTE,
        "linhas_base": linhas_base,
        "colunas_base": colunas_base,
        "N": N,
        "n0": tam["n0"],
        "n": n,
        "z": tam["z"],
        "E": 0.03,
        "pi": 0.5,
        "confianca": 0.95,
        "semente": SEMENTE,
        "commit_pre_registro": COMMIT_PRE_REGISTRO,
    }
    with open(pasta_saida / "metadados.json", "w", encoding="utf-8") as arquivo:
        json.dump(metadados, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return metadados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="caminho do CSV bruto das operações não automáticas")
    args = parser.parse_args()

    metadados = preparar(args.csv)
    print(json.dumps(metadados, ensure_ascii=False, indent=2))
    print(f"Arquivos gravados em {PASTA_DADOS}")


if __name__ == "__main__":
    main()
