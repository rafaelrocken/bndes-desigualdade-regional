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


def filtrar_subcreditos(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só os subcréditos do estudo: reembolsáveis, de UF válida, contratados de 2002 a 2025.

    Fora ficam o apoio não reembolsável, as operações interestaduais (`IE`, que não
    pertencem a nenhuma região) e 2026, ano ainda em curso na base.
    """
    mascara = (
        df["modalidade_de_apoio"].str.startswith("REEMB")
        & df["uf"].isin(UFS_REGIAO)
        & df["ano"].between(2002, 2025)
    )
    return df[mascara]


def _media_ponderada(sub: pd.DataFrame, coluna: str) -> pd.Series:
    """Média de `coluna` ponderada pelo valor contratado, por contrato (NaN se o peso for zero)."""
    peso = sub["valor_contratado_reais"]
    por_contrato = sub["numero_do_contrato"]
    numerador = (sub[coluna] * peso).groupby(por_contrato).sum()
    denominador = peso.groupby(por_contrato).sum()
    return numerador / denominador.where(denominador > 0)


def agregar_contratos(sub: pd.DataFrame) -> pd.DataFrame:
    """Transforma subcréditos em contratos: uma linha por `numero_do_contrato`.

    Regras (desenho, seção 5):
    - `uf`: a UF com a maior soma de valor no contrato; `regiao` vem dela.
    - `grupo`: "N+NE" se a região for N ou NE, senão "Demais".
    - `valor_contratado`: soma dos subcréditos.
    - `spread`: média de `juros` ponderada pelo valor, só entre os subcréditos com
      indexador em `INDEXADORES_REAIS`; NaN se o contrato não tem nenhum.
    - `prazo_amortizacao`: média ponderada pelo valor, entre todos os subcréditos.
    - `porte`, `setor`, `data`, `ano`, `cnpj`: os do subcrédito de maior valor.

    Empate (duas UFs com a mesma soma, ou dois subcréditos com o mesmo valor maior)
    fica com o primeiro na ordem original das linhas. O resultado sai ordenado
    por `numero_do_contrato`.
    """
    sub = sub.reset_index(drop=True)

    # UF com a maior soma: ordenação estável por valor decrescente preserva a ordem
    # de aparição da UF dentro do contrato, e o primeiro de cada contrato vence.
    por_uf = (
        sub.groupby(["numero_do_contrato", "uf"], sort=False)["valor_contratado_reais"]
        .sum()
        .reset_index()
        .sort_values("valor_contratado_reais", ascending=False, kind="stable")
        .drop_duplicates("numero_do_contrato", keep="first")
        .set_index("numero_do_contrato")["uf"]
    )

    # Subcrédito de maior valor, com o mesmo critério de desempate.
    maior = (
        sub.sort_values("valor_contratado_reais", ascending=False, kind="stable")
        .drop_duplicates("numero_do_contrato", keep="first")
        .set_index("numero_do_contrato")
    )

    em_reais = sub[sub["custo_financeiro"].isin(INDEXADORES_REAIS)]
    contratos = pd.DataFrame(
        {
            "uf": por_uf,
            "valor_contratado": sub.groupby("numero_do_contrato")["valor_contratado_reais"].sum(),
            "spread": _media_ponderada(em_reais, "juros"),
            "prazo_amortizacao": _media_ponderada(sub, "prazo_amortizacao_meses"),
            "porte": maior["porte_do_cliente"],
            "setor": maior["setor_bndes"],
            "data": maior["data_da_contratacao"],
            "ano": maior["ano"],
            "cnpj": maior["cnpj"],
            "n_subcreditos": sub.groupby("numero_do_contrato").size(),
        }
    )
    contratos["regiao"] = contratos["uf"].map(UFS_REGIAO)
    contratos["grupo"] = contratos["regiao"].isin(["N", "NE"]).map({True: "N+NE", False: "Demais"})
    contratos.index.name = "numero_do_contrato"
    colunas = [
        "numero_do_contrato", "uf", "regiao", "grupo", "valor_contratado", "spread",
        "prazo_amortizacao", "porte", "setor", "data", "ano", "cnpj", "n_subcreditos",
    ]
    return contratos.sort_index().reset_index()[colunas]


def montar_populacao(caminho_csv: Path) -> pd.DataFrame:
    """Atalho: lê o CSV, filtra e agrega. Devolve a população do estudo, um contrato por linha."""
    return agregar_contratos(filtrar_subcreditos(ler_csv_bndes(caminho_csv)))
