"""Gráficos do trabalho final da IA003: o BNDES compensa a desigualdade regional?

Construtores de gráficos Altair usados pelas páginas do Streamlit. Este módulo não
depende do Streamlit.

Regras da disciplina que valem para todos os gráficos:

- barras para variável qualitativa, da maior para a menor, começando em zero;
  histograma ou boxplot para variável quantitativa; nada de pizza nem 3D;
- título, fonte e legenda em todos (a fonte vai no subtítulo do título);
- os painéis da população e da amostra usam a mesma escala: o domínio dos eixos
  é calculado na população (`dominio`) e passado aos construtores, e o eixo
  vertical dos histogramas é fixado com `max_proporcao_histograma` nos dois;
- as cores dos grupos (`CORES`) nunca trocam de lugar: toda escala de cor por
  grupo é montada pela mesma função, com o domínio na ordem N+NE, Demais.

Os construtores resumem os dados no pandas antes de entregá-los ao Altair. Assim o
gráfico leva dezenas de linhas, não os 6.320 contratos, e não esbarra no limite de
5.000 linhas do Altair.
"""
import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

FONTE = "Fonte: BNDES, operações não automáticas (dados abertos, baixado em 12/11/2026)"

# Paleta Okabe-Ito (segura para daltônicos): vermelhão para N+NE, azul para o resto.
CORES: dict[str, str] = {"N+NE": "#D55E00", "Demais": "#0072B2"}
GRUPOS: list[str] = list(CORES)  # ordem fixa: N+NE, Demais

REGIOES: list[str] = ["N", "NE", "CO", "SE", "S"]
# Mesma regra de `calculo.agregar_contratos`: N e NE formam o grupo N+NE.
GRUPO_DA_REGIAO: dict[str, str] = {
    "N": "N+NE",
    "NE": "N+NE",
    "CO": "Demais",
    "SE": "Demais",
    "S": "Demais",
}

# Título dos eixos das variáveis quantitativas, com a unidade.
ROTULOS: dict[str, str] = {
    "spread": "Spread (% a.a.)",
    "prazo_amortizacao": "Prazo de amortização (meses)",
    "valor_contratado": "Valor contratado (R$)",
}

# Cinza médio: legível em fundo claro e escuro.
_CINZA = "#888888"
# Cores de apoio fora da paleta dos grupos, para não sugerir grupo onde não há.
_VERDE = "#009E73"
_ROXO = "#CC79A7"

_LARGURA = 460
_MAX_PONTOS_QQ = 2000


def _titulo(texto: str, *legenda: str) -> alt.Title:
    """Título com a fonte no subtítulo; `legenda` entra numa linha acima da fonte."""
    subtitulo = [*legenda, FONTE] if legenda else FONTE
    return alt.Title(text=texto, subtitle=subtitulo, anchor="start")


def _cor_grupo() -> alt.Color:
    """Cor por grupo, com a escala e a legenda iguais em todos os gráficos."""
    return alt.Color(
        "grupo:N",
        scale=alt.Scale(domain=GRUPOS, range=[CORES[g] for g in GRUPOS]),
        legend=alt.Legend(title="Grupo de regiões"),
    )


def _rotulo(coluna: str, rotulo: str | None) -> str:
    return rotulo or ROTULOS.get(coluna, coluna)


def dominio(pop: pd.DataFrame, coluna: str) -> tuple[float, float]:
    """Domínio de eixo da `coluna`, calculado na população: mínimo e máximo sem os
    ausentes, com 2% de folga em cada lado. Se o mínimo é zero ou positivo, o piso
    nunca fica abaixo de zero.
    """
    valores = pop[coluna].dropna()
    if valores.empty:
        raise ValueError(f"a coluna {coluna!r} não tem valores")
    menor, maior = float(valores.min()), float(valores.max())
    folga = 0.02 * (maior - menor)
    if folga == 0:  # coluna constante
        folga = 0.02 * abs(menor) or 1.0
    inferior = menor - folga
    if menor >= 0:
        inferior = max(inferior, 0.0)
    return inferior, maior + folga


def barras_regiao(df: pd.DataFrame, medida: str, titulo: str) -> alt.Chart:
    """Participação de cada região, de 0 a 100%, da maior para a menor (de cima
    para baixo), em barras horizontais.

    `medida="contratos"` conta contratos; `medida="valor"` soma `valor_contratado`.
    As cinco regiões aparecem sempre (a que não tem contrato fica em zero), e cada
    barra leva a cor do grupo da região. O eixo vai de 0 a 100% nas duas medidas,
    o que deixa os dois painéis na mesma escala.
    """
    if medida == "contratos":
        total_por_regiao = df.groupby("regiao").size().astype(float)
        eixo = "Participação nos contratos (%)"
    elif medida == "valor":
        total_por_regiao = df.groupby("regiao")["valor_contratado"].sum()
        eixo = "Participação no valor contratado (%)"
    else:
        raise ValueError(f"medida deve ser 'contratos' ou 'valor', e não {medida!r}")

    total_por_regiao = total_por_regiao.reindex(REGIOES, fill_value=0.0)
    soma = total_por_regiao.sum()
    participacao = total_por_regiao / soma if soma else total_por_regiao
    dados = pd.DataFrame(
        {
            "regiao": REGIOES,
            "grupo": [GRUPO_DA_REGIAO[r] for r in REGIOES],
            "participacao": participacao.to_numpy(),
        }
    )
    ordem = list(dados.sort_values("participacao", ascending=False, kind="stable")["regiao"])

    return (
        alt.Chart(dados, title=_titulo(titulo))
        .mark_bar()
        .encode(
            y=alt.Y("regiao:N", sort=ordem, title="Região"),
            x=alt.X(
                "participacao:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title=eixo),
            ),
            color=_cor_grupo(),
            tooltip=[
                alt.Tooltip("regiao:N", title="Região"),
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("participacao:Q", title="Participação", format=".1%"),
            ],
        )
        .properties(width=_LARGURA, height=240)
    )


def barras_setor(df: pd.DataFrame, titulo: str) -> alt.Chart:
    """Composição por setor dentro de cada grupo: a participação de cada setor nos
    contratos do grupo (soma 1 em cada grupo), em barras agrupadas.

    Os setores seguem a participação geral, do maior para o menor, de cima para
    baixo. O eixo vai de 0 a 100%.
    """
    ordem = list(df["setor"].value_counts().index)
    linhas = []
    for grupo in GRUPOS:
        contagem = df.loc[df["grupo"] == grupo, "setor"].value_counts()
        total = int(contagem.sum())
        for setor in ordem:
            n = int(contagem.get(setor, 0))
            linhas.append(
                {
                    "setor": setor,
                    "grupo": grupo,
                    "contratos": n,
                    "participacao": n / total if total else 0.0,
                }
            )
    dados = pd.DataFrame(linhas)

    return (
        alt.Chart(dados, title=_titulo(titulo))
        .mark_bar()
        .encode(
            y=alt.Y("setor:N", sort=ordem, title="Setor"),
            yOffset=alt.YOffset("grupo:N", sort=GRUPOS),
            x=alt.X(
                "participacao:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="Participação nos contratos do grupo (%)"),
            ),
            color=_cor_grupo(),
            tooltip=[
                alt.Tooltip("setor:N", title="Setor"),
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("participacao:Q", title="Participação", format=".1%"),
                alt.Tooltip("contratos:Q", title="Contratos"),
            ],
        )
        .properties(width=_LARGURA, height=300)
    )


def _proporcoes_por_faixa(
    df: pd.DataFrame, coluna: str, dominio: tuple[float, float], bins: int
) -> pd.DataFrame:
    """Uma linha por grupo e faixa, com a contagem e a proporção do grupo.

    As faixas dividem o `dominio` em `bins` partes iguais. Os ausentes saem; o
    denominador da proporção é o total de valores não ausentes do grupo, de modo
    que valores fora do `dominio` ficam sem faixa mas contam no denominador. É
    daqui que saem tanto as barras do `histograma` quanto o `max_proporcao_histograma`,
    então os dois concordam por construção.
    """
    bordas = np.linspace(dominio[0], dominio[1], bins + 1)
    quadros = []
    for grupo in GRUPOS:
        x = df.loc[df["grupo"] == grupo, coluna].dropna().to_numpy(dtype=float)
        contagens, _ = np.histogram(x, bins=bordas)
        quadros.append(
            pd.DataFrame(
                {
                    "grupo": grupo,
                    "inicio": bordas[:-1],
                    "fim": bordas[1:],
                    "contratos": contagens,
                    "proporcao": contagens / x.size if x.size else np.zeros(bins),
                }
            )
        )
    return pd.concat(quadros, ignore_index=True)


def max_proporcao_histograma(
    dfs: list[pd.DataFrame], coluna: str, dominio: tuple[float, float], bins: int = 30
) -> float:
    """Maior proporção de uma faixa dentro de um grupo, entre todos os `dfs`, com
    5% de folga (no máximo 1). Serve de `max_y` para os histogramas da população e
    da amostra, que assim ficam com o mesmo eixo vertical.

    Usa as mesmas faixas do `histograma` (mesmos `dominio` e `bins`).
    """
    maiores = [_proporcoes_por_faixa(df, coluna, dominio, bins)["proporcao"].max() for df in dfs]
    maior = max(maiores, default=0.0)
    if not maior > 0:
        raise ValueError(f"nenhum valor de {coluna!r} dentro do domínio para medir a proporção")
    return float(min(1.0, 1.05 * maior))


def histograma(
    df: pd.DataFrame,
    coluna: str,
    dominio: tuple[float, float],
    titulo: str,
    bins: int = 30,
    *,
    rotulo: str | None = None,
    max_y: float | None = None,
) -> alt.FacetChart:
    """Histograma de `coluna` com um quadro por grupo, um acima do outro.

    Os ausentes saem. As faixas dividem o `dominio` em `bins` partes iguais, as
    mesmas para os dois grupos e para os dois painéis. A altura é a proporção dos
    contratos do grupo em cada faixa (não a contagem), que deixa população e
    amostra comparáveis apesar dos tamanhos diferentes; o denominador é o total de
    valores não ausentes do grupo, de modo que valores fora do `dominio` ficam sem
    barra mas contam no denominador. O eixo x usa `dominio` com `clamp`.

    `rotulo` troca o título do eixo x (por padrão vem de `ROTULOS`). `max_y` fixa o
    eixo y em [0, max_y]; para comparar população e amostra, passe nos dois
    gráficos o valor de `max_proporcao_histograma([pop, amostra], ...)`. Sem
    `max_y`, o eixo y se ajusta ao próprio gráfico.
    """
    dados = _proporcoes_por_faixa(df, coluna, dominio, bins)
    escala_y = alt.Scale(zero=True) if max_y is None else alt.Scale(domain=[0, max_y], clamp=True)

    eixo_x = _rotulo(coluna, rotulo)
    return (
        alt.Chart(dados)
        .mark_bar()
        .encode(
            x=alt.X(
                "inicio:Q",
                bin="binned",
                scale=alt.Scale(domain=list(dominio), clamp=True),
                title=eixo_x,
            ),
            x2="fim:Q",
            y=alt.Y(
                "proporcao:Q",
                scale=escala_y,
                axis=alt.Axis(format=".0%"),
                # duas linhas: o título se repete em cada quadro e numa linha só não cabe
                title=["Proporção dos contratos", "do grupo em cada faixa (%)"],
            ),
            color=_cor_grupo(),
            tooltip=[
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("inicio:Q", title="Faixa: de", format=",.3~g"),
                alt.Tooltip("fim:Q", title="até", format=",.3~g"),
                alt.Tooltip("contratos:Q", title="Contratos"),
                alt.Tooltip("proporcao:Q", title="Proporção do grupo", format=".1%"),
            ],
        )
        .properties(width=_LARGURA, height=150)
        .facet(
            row=alt.Row(
                "grupo:N",
                sort=GRUPOS,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12),
            ),
        )
        .properties(title=_titulo(titulo))
    )


def _caixa(x: np.ndarray) -> tuple[dict, np.ndarray]:
    """Resumo de cinco números de Tukey e os pontos fora dos bigodes.

    Os bigodes vão até o dado mais extremo que ainda cai a 1,5 amplitude
    interquartil da caixa (a regra do `mark_boxplot` do Altair e do R).
    """
    q1, mediana, q3 = np.percentile(x, [25, 50, 75])
    amplitude = q3 - q1
    dentro = (x >= q1 - 1.5 * amplitude) & (x <= q3 + 1.5 * amplitude)
    resumo = {
        "inferior": float(x[dentro].min()),
        "q1": float(q1),
        "mediana": float(mediana),
        "q3": float(q3),
        "superior": float(x[dentro].max()),
        "contratos": int(x.size),
    }
    return resumo, x[~dentro]


def boxplot(
    df: pd.DataFrame,
    coluna: str,
    dominio: tuple[float, float],
    titulo: str,
    *,
    rotulo: str | None = None,
) -> alt.LayerChart:
    """Boxplot horizontal de `coluna`, uma caixa por grupo, no mesmo `dominio`.

    Caixa do primeiro ao terceiro quartil, traço branco na mediana, bigodes de
    Tukey (1,5 amplitude interquartil) e os pontos fora deles. Os ausentes saem.
    `rotulo` troca o título do eixo x (por padrão vem de `ROTULOS`).
    """
    caixas, fora = [], []
    for grupo in GRUPOS:
        x = df.loc[df["grupo"] == grupo, coluna].dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue
        resumo, extremos = _caixa(x)
        caixas.append({"grupo": grupo, **resumo})
        fora += [{"grupo": grupo, "valor": float(v)} for v in extremos]
    caixas = pd.DataFrame(caixas)

    escala = alt.Scale(domain=list(dominio), clamp=True)
    eixo_x = _rotulo(coluna, rotulo)
    y = alt.Y("grupo:N", sort=GRUPOS, title="Grupo")
    cor = _cor_grupo()

    bigodes = (
        alt.Chart(caixas)
        .mark_rule(strokeWidth=2)
        .encode(x=alt.X("inferior:Q", scale=escala, title=eixo_x), x2="superior:Q", y=y, color=cor)
    )
    corpo = (
        alt.Chart(caixas)
        .mark_bar(size=34, opacity=0.9)
        .encode(
            x=alt.X("q1:Q", scale=escala, title=eixo_x),
            x2="q3:Q",
            y=y,
            color=cor,
            tooltip=[
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("inferior:Q", title="Bigode inferior", format=",.3~g"),
                alt.Tooltip("q1:Q", title="1º quartil", format=",.3~g"),
                alt.Tooltip("mediana:Q", title="Mediana", format=",.3~g"),
                alt.Tooltip("q3:Q", title="3º quartil", format=",.3~g"),
                alt.Tooltip("superior:Q", title="Bigode superior", format=",.3~g"),
                alt.Tooltip("contratos:Q", title="Contratos"),
            ],
        )
    )
    traco = (
        alt.Chart(caixas)
        .mark_tick(color="white", thickness=3, size=34, orient="vertical")
        .encode(x=alt.X("mediana:Q", scale=escala, title=eixo_x), y=y)
    )
    camadas = [bigodes, corpo, traco]
    if fora:
        pontos = (
            alt.Chart(pd.DataFrame(fora))
            .mark_point(filled=True, size=18, opacity=0.5)
            .encode(x=alt.X("valor:Q", scale=escala, title=eixo_x), y=y, color=cor)
        )
        camadas.append(pontos)

    return alt.layer(*camadas).properties(
        title=_titulo(titulo), width=_LARGURA, height=170
    )


def qqplot(x, titulo: str) -> alt.LayerChart:
    """Gráfico quantil-quantil contra a normal: quantis teóricos no eixo x, valores
    ordenados no eixo y, e a reta de referência da normal ajustada aos dados.

    Os ausentes saem. Acima de 2.000 valores, o gráfico leva 2.000 pontos igualmente
    espaçados na ordenação (com os extremos), para caber no limite de linhas do
    Altair; a reta usa todos os valores.
    """
    valores = np.asarray(x, dtype=float)
    valores = valores[~np.isnan(valores)]
    if valores.size < 3:
        raise ValueError("qqplot precisa de pelo menos 3 valores não ausentes")
    (teorico, observado), (inclinacao, intercepto, _) = stats.probplot(valores, dist="norm")
    if teorico.size > _MAX_PONTOS_QQ:
        indices = np.unique(np.linspace(0, teorico.size - 1, _MAX_PONTOS_QQ).round().astype(int))
        teorico, observado = teorico[indices], observado[indices]

    nome_pontos = "Valores observados"
    nome_reta = "Reta da normal (referência)"
    pontos = pd.DataFrame({"teorico": teorico, "observado": observado, "serie": nome_pontos})
    extremos = np.array([teorico.min(), teorico.max()])
    reta = pd.DataFrame(
        {"teorico": extremos, "observado": intercepto + inclinacao * extremos, "serie": nome_reta}
    )

    x_enc = alt.X("teorico:Q", scale=alt.Scale(zero=False), title="Quantis teóricos da normal padrão")
    y_enc = alt.Y("observado:Q", scale=alt.Scale(zero=False), title="Valores observados (ordenados)")
    cor = alt.Color(
        "serie:N",
        scale=alt.Scale(domain=[nome_pontos, nome_reta], range=[_CINZA, _VERDE]),
        legend=alt.Legend(title="Legenda"),
    )
    camada_pontos = (
        alt.Chart(pontos)
        .mark_point(filled=True, size=25, opacity=0.6)
        .encode(x=x_enc, y=y_enc, color=cor)
    )
    camada_reta = alt.Chart(reta).mark_line(strokeWidth=2).encode(x=x_enc, y=y_enc, color=cor)
    return alt.layer(camada_pontos, camada_reta).properties(
        title=_titulo(titulo), width=_LARGURA, height=320
    )


def intervalos(resultados: list[dict], titulo: str) -> alt.FacetChart:
    """Intervalo de confiança de cada hipótese, com o valor verdadeiro da população.

    `resultados` é uma lista de dicionários com `id` (ex.: "H3"), `estimativa`,
    `li`, `ls`, `verdadeiro` e `unidade`. Cada item ocupa uma linha, com escala x
    própria (as unidades diferem): a linha vai de `li` a `ls`, o ponto marca a
    estimativa da amostra e o losango marca o valor verdadeiro da população.
    """
    if not resultados:
        raise ValueError("resultados está vazio: passe ao menos um item para desenhar")
    dados = pd.DataFrame(resultados)[["id", "unidade", "estimativa", "li", "ls", "verdadeiro"]]
    dados["rotulo"] = dados["id"] + " (" + dados["unidade"] + ")"

    nome_estimativa = "Estimativa da amostra (ponto)"
    nome_verdadeiro = "Valor verdadeiro da população (losango)"
    escala = alt.Scale(zero=False)
    eixo = alt.Axis(title=None, tickCount=5, labelOverlap="greedy")  # a unidade vai no rótulo da linha

    base = alt.Chart(dados)
    faixa = base.mark_rule(strokeWidth=3, color=_CINZA).encode(
        x=alt.X("li:Q", scale=escala, axis=eixo),
        x2="ls:Q",
        tooltip=[
            alt.Tooltip("rotulo:N", title="Hipótese"),
            alt.Tooltip("li:Q", title="Limite inferior", format=",.4~g"),
            alt.Tooltip("ls:Q", title="Limite superior", format=",.4~g"),
        ],
    )
    marcas = (
        base.transform_fold(["estimativa", "verdadeiro"], as_=["origem", "valor"])
        .transform_calculate(
            tipo=f"datum.origem == 'estimativa' ? '{nome_estimativa}' : '{nome_verdadeiro}'"
        )
        .mark_point(filled=True, opacity=1)
        .encode(
            x=alt.X("valor:Q", scale=escala, axis=eixo),
            shape=alt.Shape(
                "tipo:N",
                scale=alt.Scale(domain=[nome_estimativa, nome_verdadeiro], range=["circle", "diamond"]),
                legend=alt.Legend(title="Marcas", labelLimit=320),
            ),
            color=alt.Color(
                "tipo:N",
                scale=alt.Scale(domain=[nome_estimativa, nome_verdadeiro], range=[_VERDE, _ROXO]),
                legend=alt.Legend(title="Marcas", labelLimit=320),
            ),
            size=alt.Size(
                "tipo:N",
                scale=alt.Scale(domain=[nome_estimativa, nome_verdadeiro], range=[70, 170]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("rotulo:N", title="Hipótese"),
                alt.Tooltip("tipo:N", title="Marca"),
                alt.Tooltip("valor:Q", title="Valor", format=",.4~g"),
            ],
        )
    )
    legenda = (
        "Linha = intervalo de confiança; ponto = estimativa da amostra; "
        "losango = valor verdadeiro da população."
    )
    return (
        alt.layer(faixa, marcas)
        .properties(width=_LARGURA, height=50)
        .facet(
            row=alt.Row(
                "rotulo:N",
                sort=list(dados["rotulo"]),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12),
            ),
        )
        .resolve_scale(x="independent")
        .properties(title=_titulo(titulo, legenda))
    )
