"""Cálculo puro do trabalho final da IA003: o BNDES compensa a desigualdade regional?

Este módulo não depende do Streamlit. Ele lê a base de operações não automáticas
do BNDES (dados abertos) e, nas próximas etapas, agrega subcréditos em contratos,
sorteia a amostra e roda os testes de hipóteses.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


# --- Amostra e inferência -----------------------------------------------------------
# Nenhuma função abaixo imprime nem lê arquivo: recebem números e devolvem números.

ALTERNATIVAS = ("bilateral", "menor")


def _validar_alternativa(alternativa: str) -> None:
    """Aceita só "bilateral" ou "menor" (H1 < H0 à esquerda); qualquer outro valor é erro."""
    if alternativa not in ALTERNATIVAS:
        raise ValueError(f'alternativa deve ser "bilateral" ou "menor", não {alternativa!r}')


def _sem_nan(x) -> np.ndarray:
    """Vetor de floats sem os NaN, a partir de lista, array ou Series."""
    v = np.asarray(x, dtype=float)
    return v[~np.isnan(v)]


def tamanho_amostra(N: int, E: float = 0.03, confianca: float = 0.95, pi: float = 0.5) -> dict:
    """Tamanho da amostra pela fórmula usual para proporção, com correção para população finita.

    n0 = z² π (1 − π) / E²  e  n = n0 / (1 + n0 / N), arredondado para cima.
    Devolve `z`, `n0` (sem arredondar) e `n` (inteiro).
    """
    z = float(stats.norm.ppf(1 - (1 - confianca) / 2))
    n0 = z**2 * pi * (1 - pi) / E**2
    n = math.ceil(n0 / (1 + n0 / N))
    return {"z": z, "n0": n0, "n": n}


def sortear_amostra(pop: pd.DataFrame, n: int = 913, semente: int = 20261112) -> pd.DataFrame:
    """Amostra aleatória simples, sem reposição, reprodutível pela semente.

    A população é ordenada por `numero_do_contrato` antes do sorteio, então a ordem
    em que as linhas chegam não muda quem é sorteado.
    """
    return pop.sort_values("numero_do_contrato").sample(n=n, replace=False, random_state=semente)


def descritiva(s: pd.Series) -> pd.Series:
    """Estatísticas descritivas de uma variável, ignorando NaN.

    Índice: n, media, mediana, desvio_padrao (ddof=1), q1, q3, minimo, maximo.
    Os quartis usam o `quantile` padrão do pandas (interpolação linear).
    """
    v = pd.Series(_sem_nan(s))
    return pd.Series(
        {
            "n": float(len(v)),
            "media": v.mean(),
            "mediana": v.median(),
            "desvio_padrao": v.std(ddof=1),
            "q1": v.quantile(0.25),
            "q3": v.quantile(0.75),
            "minimo": v.min(),
            "maximo": v.max(),
        }
    )


def ic_media(x, confianca: float = 0.95) -> tuple[float, float, float]:
    """Intervalo de confiança t para a média, ignorando NaN: (média, limite inferior, limite superior).

    Usa o desvio padrão com ddof=1 e a distribuição t com n − 1 graus de liberdade.
    """
    v = _sem_nan(x)
    n = v.size
    media = float(v.mean())
    ep = float(v.std(ddof=1)) / math.sqrt(n)
    t_crit = float(stats.t.ppf(1 - (1 - confianca) / 2, n - 1))
    return media, media - t_crit * ep, media + t_crit * ep


def teste_z_proporcao(x: int, n: int, p0: float, alfa: float = 0.05) -> dict:
    """Teste Z bilateral para uma proporção, H0: π = p0.

    O erro padrão do teste usa p0: ep_teste = √(p0 (1 − p0) / n). Os intervalos de
    confiança, de Wald e de Wilson, são sempre de 95% e usam p̂ (o erro padrão de
    Wald é √(p̂ (1 − p̂) / n)). `rejeita` é p_valor < alfa.
    """
    p_hat = x / n
    ep_teste = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / ep_teste
    p_valor = float(2 * stats.norm.sf(abs(z)))

    zc = float(stats.norm.ppf(0.975))
    ep_wald = math.sqrt(p_hat * (1 - p_hat) / n)
    ic_wald = (p_hat - zc * ep_wald, p_hat + zc * ep_wald)

    denominador = 1 + zc**2 / n
    centro = (p_hat + zc**2 / (2 * n)) / denominador
    meia = zc * math.sqrt(p_hat * (1 - p_hat) / n + zc**2 / (4 * n**2)) / denominador
    ic_wilson = (centro - meia, centro + meia)

    return {
        "p_hat": p_hat,
        "ep_teste": ep_teste,
        "z": z,
        "p_valor": p_valor,
        "ic_wald": ic_wald,
        "ic_wilson": ic_wilson,
        "rejeita": bool(p_valor < alfa),
    }


def teste_welch(a, b, alternativa: str = "bilateral", alfa: float = 0.05) -> dict:
    """Teste t de Welch para a diferença de médias, `a` (N+NE) menos `b` (Demais).

    Descarta NaN de cada grupo. `alternativa` é "bilateral" (H1: diferença ≠ 0) ou
    "menor" (H1: diferença < 0). O intervalo de confiança da diferença é sempre o
    bilateral de 95%, qualquer que seja a alternativa. `d_cohen` usa o desvio padrão
    combinado. `rejeita` é p_valor < alfa.
    """
    _validar_alternativa(alternativa)
    va, vb = _sem_nan(a), _sem_nan(b)
    n_a, n_b = va.size, vb.size

    bilateral = stats.ttest_ind(va, vb, equal_var=False)
    if alternativa == "bilateral":
        resultado = bilateral
    else:
        resultado = stats.ttest_ind(va, vb, equal_var=False, alternative="less")
    ic = bilateral.confidence_interval(0.95)

    media_a, media_b = float(va.mean()), float(vb.mean())
    diferenca = media_a - media_b
    dp_combinado = math.sqrt(((n_a - 1) * va.var(ddof=1) + (n_b - 1) * vb.var(ddof=1)) / (n_a + n_b - 2))
    p_valor = float(resultado.pvalue)

    return {
        "n_a": n_a,
        "n_b": n_b,
        "media_a": media_a,
        "media_b": media_b,
        "diferenca": diferenca,
        "t": float(resultado.statistic),
        "gl": float(resultado.df),
        "p_valor": p_valor,
        "ic_diferenca": (float(ic.low), float(ic.high)),
        "d_cohen": diferenca / dp_combinado,
        "rejeita": bool(p_valor < alfa),
    }


def condicoes_proporcao(n: int, p0: float) -> dict:
    """Condição do teste Z de uma proporção: n·p0 ≥ 10 e n·(1 − p0) ≥ 10."""
    np0 = n * p0
    n_1_menos_p0 = n * (1 - p0)
    return {"np0": np0, "n_1_menos_p0": n_1_menos_p0, "ok": bool(np0 >= 10 and n_1_menos_p0 >= 10)}


def condicoes_media(n_a: int, n_b: int) -> dict:
    """Condição do teste t de Welch pelo tamanho: n ≥ 30 nos dois grupos."""
    return {"n_a": n_a, "n_b": n_b, "ok": bool(n_a >= 30 and n_b >= 30)}


def poder_z_proporcao(p_verdadeiro: float, p0: float, n: int, alfa: float = 0.05) -> float:
    """Poder do teste Z bilateral de uma proporção, por aproximação normal.

    A região de rejeição usa o erro padrão sob H0 (ep0 com p0); a distribuição de p̂
    usa o erro padrão sob a verdade (ep1 com p_verdadeiro, que deve estar em (0, 1)):
    poder = Φ((p0 − z·ep0 − p) / ep1) + 1 − Φ((p0 + z·ep0 − p) / ep1), z = z de 1 − α/2.
    Com p_verdadeiro == p0 o resultado é exatamente `alfa`.
    """
    z = float(stats.norm.ppf(1 - alfa / 2))
    ep0 = math.sqrt(p0 * (1 - p0) / n)
    ep1 = math.sqrt(p_verdadeiro * (1 - p_verdadeiro) / n)
    cauda_inferior = stats.norm.cdf((p0 - z * ep0 - p_verdadeiro) / ep1)
    cauda_superior = stats.norm.sf((p0 + z * ep0 - p_verdadeiro) / ep1)
    return float(cauda_inferior + cauda_superior)


def poder_welch(
    dif_verdadeira: float, dp_a: float, dp_b: float, n_a: int, n_b: int,
    alfa: float = 0.05, alternativa: str = "bilateral",
) -> float:
    """Poder do teste de Welch por aproximação normal, com a diferença verdadeira μa − μb.

    ep = √(dp_a²/n_a + dp_b²/n_b) e δ = dif_verdadeira. Bilateral:
    Φ(−z_{α/2} − δ/ep) + 1 − Φ(z_{α/2} − δ/ep). "menor": Φ(−z_α − δ/ep).
    Com δ = 0 o resultado é exatamente `alfa`.
    """
    _validar_alternativa(alternativa)
    ep = math.sqrt(dp_a**2 / n_a + dp_b**2 / n_b)
    deslocamento = dif_verdadeira / ep
    if alternativa == "menor":
        z = float(stats.norm.ppf(1 - alfa))
        return float(stats.norm.cdf(-z - deslocamento))
    z = float(stats.norm.ppf(1 - alfa / 2))
    return float(stats.norm.cdf(-z - deslocamento) + stats.norm.sf(z - deslocamento))


def julgar_decisao(rejeita: bool, valor_verdadeiro: float, valor_h0: float, alternativa: str) -> str:
    """Compara a decisão do teste com a verdade conhecida da população.

    H0 é verdadeira quando `valor_verdadeiro == valor_h0` (bilateral) ou quando
    `valor_verdadeiro >= valor_h0` ("menor"). Rejeitar uma H0 verdadeira é "erro tipo I";
    não rejeitar uma H0 falsa é "erro tipo II"; os outros dois casos são "acerto".
    """
    _validar_alternativa(alternativa)
    if alternativa == "bilateral":
        h0_verdadeira = valor_verdadeiro == valor_h0
    else:
        h0_verdadeira = valor_verdadeiro >= valor_h0
    if rejeita:
        return "erro tipo I" if h0_verdadeira else "acerto"
    return "acerto" if h0_verdadeira else "erro tipo II"


# As cinco hipóteses, gravadas antes do sorteio (desenho, seção 7). α = 0,05.
# `variavel` é a coluna da população que cada teste usa; `p0_ou_delta0` é o valor de H0.
HIPOTESES: list[dict] = [
    {
        "id": "H1",
        "parametro": "π = proporção de contratos em N+NE",
        "h0": r"$\pi = 0{,}196$ (PIB)",
        "h1": r"$\pi \neq 0{,}196$ (bilateral)",
        "p0_ou_delta0": 0.196,
        "alternativa": "bilateral",
        "referencia": "Régua do PIB: 19,6% (IBGE, Contas Regionais 2023: NE 13,8% + N 5,8%)",
        "variavel": "grupo",
        "teste": "Z de uma proporção",
    },
    {
        "id": "H2",
        "parametro": "o mesmo π (proporção de contratos em N+NE)",
        "h0": r"$\pi = 0{,}354$ (população)",
        "h1": r"$\pi \neq 0{,}354$ (bilateral)",
        "p0_ou_delta0": 0.354,
        "alternativa": "bilateral",
        "referencia": "Régua da população: 35,4% (IBGE, Censo 2022: NE 26,9% + N 8,5%)",
        "variavel": "grupo",
        "teste": "Z de uma proporção",
    },
    {
        "id": "H3",
        "parametro": "δ = μ_NNE − μ_demais do spread médio",
        "h0": r"$\delta = 0$",
        "h1": r"$\delta \neq 0$ (bilateral)",
        "p0_ou_delta0": 0.0,
        "alternativa": "bilateral",
        "referencia": (
            "Régua das condições: Política Nacional de Desenvolvimento Regional e "
            "Estratégia Nordeste 2023 (condições diferenciadas para N/NE)"
        ),
        "variavel": "spread",
        "teste": "t de Welch",
    },
    {
        "id": "H4",
        "parametro": "o mesmo δ (μ_NNE − μ_demais do spread médio)",
        "h0": r"$\delta \ge 0$",
        "h1": r"$\delta < 0$: N+NE pagam menos, a promessa da política regional (unilateral à esquerda)",
        "p0_ou_delta0": 0.0,
        "alternativa": "menor",
        "referencia": (
            "Régua das condições: Política Nacional de Desenvolvimento Regional e "
            "Estratégia Nordeste 2023 (condições diferenciadas para N/NE)"
        ),
        "variavel": "spread",
        "teste": "t de Welch",
    },
    {
        "id": "H5",
        "parametro": "μ_NNE − μ_demais do prazo médio de amortização",
        "h0": r"$\mu_{NNE} - \mu_{demais} = 0$",
        "h1": r"$\mu_{NNE} - \mu_{demais} \neq 0$ (bilateral)",
        "p0_ou_delta0": 0.0,
        "alternativa": "bilateral",
        "referencia": (
            "Teste extra: o prazo, a outra metade da pergunta da régua das condições "
            "(N+NE pagam menos e têm mais prazo?)"
        ),
        "variavel": "prazo_amortizacao",
        "teste": "t de Welch",
    },
]
