"""Cálculo puro do trabalho final da IA003: o BNDES compensa a desigualdade regional?

Este módulo não depende do Streamlit. Ele lê a base de operações não automáticas
do BNDES (dados abertos) e, nas próximas etapas, agrega subcréditos em contratos,
sorteia a amostra e roda os testes de hipóteses.
"""
from pathlib import Path

import pandas as pd

# As 27 unidades da federação e a região de cada uma (N, NE, CO, SE, S).
UFS_REGIAO: dict[str, str] = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "N"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "NE"),
    **dict.fromkeys(["DF", "GO", "MS", "MT"], "CO"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "SE"),
    **dict.fromkeys(["PR", "RS", "SC"], "S"),
}

# Indexadores em reais: só os subcréditos com esses custos financeiros entram no spread.
INDEXADORES_REAIS: frozenset[str] = frozenset(
    {
        "TJLP",
        "TJ462",
        "TJ453",
        "TJ6",
        "TJ3",
        "TLP",
        "SELIC",
        "IPCA",
        "TAXA REFERENCIAL (TR)",
        "CDI",
    }
)

# Colunas que chegam como texto com vírgula decimal e viram número.
COLUNAS_NUMERICAS: tuple[str, ...] = (
    "valor_contratado_reais",
    "valor_desembolsado_reais",
    "juros",
    "prazo_carencia_meses",
    "prazo_amortizacao_meses",
)


def ler_csv_bndes(caminho: Path) -> pd.DataFrame:
    """Lê o CSV bruto do BNDES e devolve a base limpa, com uma linha por subcrédito.

    O arquivo vem em latin-1, separado por ponto e vírgula, com vírgula decimal e
    textos preenchidos com espaços à direita. Tudo é lido como texto, os espaços
    são removidos, as colunas numéricas viram número e a data de contratação vira
    data. Acrescenta a coluna `ano`, tirada da data.
    """
    df = pd.read_csv(caminho, sep=";", encoding="latin-1", dtype=str)
    df = df.apply(lambda coluna: coluna.str.strip())
    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna].str.replace(",", ".", regex=False))
    df["data_da_contratacao"] = pd.to_datetime(df["data_da_contratacao"], format="%Y-%m-%d")
    df["ano"] = df["data_da_contratacao"].dt.year
    return df
