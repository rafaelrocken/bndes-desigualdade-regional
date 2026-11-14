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

# Nomes por extenso para a tela (os dados guardam as siglas e os rótulos curtos).
NOME_GRUPO: dict[str, str] = {"N+NE": "Norte e Nordeste", "Demais": "Demais regiões"}
NOME_REGIAO: dict[str, str] = {
    "N": "Norte",
    "NE": "Nordeste",
    "CO": "Centro-Oeste",
    "SE": "Sudeste",
    "S": "Sul",
}
NOME_SETOR: dict[str, str] = {
    "INFRAESTRUTURA": "Infraestrutura",
    "INDUSTRIA": "Indústria",
    "COMERCIO/SERVICOS": "Comércio e serviços",
    "AGROPECUÁRIA": "Agropecuária",
}
NOME_PORTE: dict[str, str] = {"GRANDE": "Grande", "MÉDIA": "Média", "PEQUENA": "Pequena", "MICRO": "Micro"}

FONTE_IBGE = "IBGE: PIB das Contas Regionais 2023 e população do Censo 2022 (SIDRA)"

# Limiares do mapa (proporção dos contratos). São constantes: a escala de cor do mapa é
# a mesma nos Painéis 1 e 2, e no modo Região e no modo Estado, por construção.
LIMIARES_MAPA: list[float] = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
# Rampa sequencial de um só matiz (azul-ardósia, pouco saturado), clara -> escura, uma cor por
# classe, até #1f3550, perto da cor primária do tema (#3B4B63): casa com a tela. Fica longe do
# azul saturado do grupo "Demais regiões" (#0072B2) e do vermelhão do Norte e Nordeste, para o
# mapa não sugerir grupo onde a cor mede tamanho.
RAMPA_MAPA: list[str] = ["#eef1f5", "#d4dbe5", "#b3bfcf", "#8d9db4", "#677c98", "#43597a", "#1f3550"]

# Cores das réguas: os contratos levam a cor do grupo; PIB e população, dois cinzas.
_CINZA_PIB = "#7b7b7b"
_CINZA_POP = "#c2c2c2"

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


# Expressão Vega que troca o rótulo curto do grupo pelo nome por extenso na legenda.
_ROTULO_GRUPO = (
    f"datum.label == 'N+NE' ? '{NOME_GRUPO['N+NE']}' : "
    f"(datum.label == 'Demais' ? '{NOME_GRUPO['Demais']}' : datum.label)"
)


# No cabeçalho das facetas o Vega expõe o valor em `datum.value`, não em `datum.label`.
_ROTULO_GRUPO_CABECALHO = (
    f"datum.value == 'N+NE' ? '{NOME_GRUPO['N+NE']}' : "
    f"(datum.value == 'Demais' ? '{NOME_GRUPO['Demais']}' : datum.value)"
)


# Rótulo de eixo numérico no padrão brasileiro: troca o ponto decimal pela vírgula (e a
# vírgula de milhar pelo ponto). O Vega formata com o ponto americano.
_NUMERO_BR = "replace(replace(replace(datum.label, ',', '#'), '.', ','), '#', '.')"


def _eixo(**opcoes) -> alt.Axis:
    """Eixo numérico com rótulos em português (vírgula decimal). O título vai no canal."""
    return alt.Axis(labelExpr=_NUMERO_BR, **opcoes)


def _br(x: float, casas: int = 1) -> str:
    """Número com vírgula decimal e ponto de milhar (o Vega formataria com ponto)."""
    texto = f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "−" + texto[1:] if texto.startswith("-") else texto


def _pct(x: float, casas: int = 1) -> str:
    return _br(100 * x, casas) + "%"


def _cor_grupo(posicao: str = "right") -> alt.Color:
    """Cor por grupo, com a escala e a legenda iguais em todos os gráficos. `posicao` põe a
    legenda à direita (padrão) ou no topo, quando o gráfico divide a linha com outro."""
    opcoes = {"orient": "top", "direction": "horizontal"} if posicao == "top" else {}
    return alt.Color(
        "grupo:N",
        scale=alt.Scale(domain=GRUPOS, range=[CORES[g] for g in GRUPOS]),
        legend=alt.Legend(title="Grupo de regiões", labelExpr=_ROTULO_GRUPO, **opcoes),
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
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("regiao:N", sort=ordem, title="Região"),
            x=alt.X(
                "participacao:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title=eixo, tickCount=8),
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


def barras_setor(df: pd.DataFrame, titulo: str, *, legenda: tuple[str, ...] = ()) -> alt.Chart:
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
        alt.Chart(dados, title=_titulo(titulo, *legenda))
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("setor:N", sort=ordem, title=None, axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("grupo:N", sort=GRUPOS),
            x=alt.X(
                "participacao:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", title="Participação nos contratos do grupo (%)", tickCount=10),
            ),
            color=_cor_grupo(),
            tooltip=[
                alt.Tooltip("setor:N", title="Setor"),
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("participacao:Q", title="Participação", format=".1%"),
                alt.Tooltip("contratos:Q", title="Contratos"),
            ],
        )
        .properties(width=_LARGURA, height=420)
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
    legenda: tuple[str, ...] = (),
    largura: int = _LARGURA,
) -> alt.FacetChart:
    """Histograma de `coluna` com um quadro por grupo, um acima do outro.

    Os ausentes saem. As faixas dividem o `dominio` em `bins` partes iguais, as
    mesmas para os dois grupos e para os dois painéis. A altura é a proporção dos
    contratos do grupo em cada faixa (não a contagem), que deixa população e
    amostra comparáveis apesar dos tamanhos diferentes; o denominador é o total de
    valores não ausentes do grupo, de modo que valores fora do `dominio` ficam sem
    barra mas contam no denominador. O eixo x usa `dominio` com `clamp`.

    `legenda` entra no subtítulo, acima da fonte; `largura` é a de cada quadro.
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
                title=eixo_x, axis=_eixo(),
            ),
            x2="fim:Q",
            y=alt.Y(
                "proporcao:Q",
                scale=escala_y,
                axis=alt.Axis(format=".0%"),
                # duas linhas: o título se repete em cada quadro e numa linha só não cabe
                title=["Proporção dos", "contratos do grupo"],
            ),
            color=_cor_grupo("top"),
            tooltip=[
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("inicio:Q", title="Faixa: de", format=",.3~g"),
                alt.Tooltip("fim:Q", title="até", format=",.3~g"),
                alt.Tooltip("contratos:Q", title="Contratos"),
                alt.Tooltip("proporcao:Q", title="Proporção do grupo", format=".1%"),
            ],
        )
        .properties(width=largura, height=140)
        .facet(
            row=alt.Row(
                "grupo:N",
                sort=GRUPOS,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=13, labelExpr=_ROTULO_GRUPO_CABECALHO),
            ),
        )
        .properties(title=_titulo(titulo, *legenda))
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
    legenda: tuple[str, ...] = (),
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
    y = alt.Y("grupo:N", sort=GRUPOS, title=None, axis=alt.Axis(labelExpr=_ROTULO_GRUPO, labelLimit=160))
    cor = _cor_grupo()

    bigodes = (
        alt.Chart(caixas)
        .mark_rule(strokeWidth=2)
        .encode(x=alt.X("inferior:Q", scale=escala, title=eixo_x, axis=_eixo()), x2="superior:Q", y=y, color=cor)
    )
    corpo = (
        alt.Chart(caixas)
        .mark_bar(size=48, opacity=0.9, cornerRadius=3)
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
        .mark_tick(color="white", thickness=3, size=48, orient="vertical")
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
        title=_titulo(titulo, *legenda), width=_LARGURA, height=330
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

    x_enc = alt.X("teorico:Q", scale=alt.Scale(zero=False), title="Quantis teóricos da normal padrão", axis=_eixo())
    y_enc = alt.Y("observado:Q", scale=alt.Scale(zero=False), title="Valores observados (ordenados)", axis=_eixo())
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
        title=_titulo(titulo), width=_LARGURA, height=380
    )


def distribuicao_p_chapeu(
    p_chapeus, referencias: dict[str, float], titulo: str, fonte: str, bins: int = 25
) -> alt.LayerChart:
    """Histograma das proporções amostrais (p̂, de 0 a 1) de muitos sorteios, em %, com
    linhas verticais de referência.

    `referencias` leva, na ordem da legenda, o nome e o valor (de 0 a 1) de até três
    linhas: a primeira em roxo (o valor verdadeiro, como em `intervalos`), a segunda em
    verde (a estimativa da amostra) e a terceira em cinza-escuro tracejado. As barras
    contam sorteios e começam em zero. `fonte` substitui a fonte padrão no subtítulo,
    porque os dados são sorteios da população, não a base direto.
    """
    valores = 100 * np.asarray(p_chapeus, dtype=float)
    if valores.size == 0:
        raise ValueError("p_chapeus está vazio")
    if not 1 <= len(referencias) <= 3:
        raise ValueError("passe de uma a três referências")
    contagens, bordas = np.histogram(valores, bins=bins)
    barras = pd.DataFrame({"inicio": bordas[:-1], "fim": bordas[1:], "sorteios": contagens})
    nomes = list(referencias)
    linhas = pd.DataFrame({"referencia": nomes, "valor": [100 * v for v in referencias.values()]})
    linhas["tracejado"] = [i == 2 for i in range(len(nomes))]

    eixo_x = "p̂ = contratos do Norte e Nordeste na amostra (%)"
    escala_x = alt.Scale(zero=False)
    camada_barras = (
        alt.Chart(barras)
        .mark_bar(color="#BBBBBB")
        .encode(
            x=alt.X("inicio:Q", bin="binned", scale=escala_x, title=eixo_x, axis=_eixo()),
            x2="fim:Q",
            y=alt.Y("sorteios:Q", scale=alt.Scale(zero=True), title="Número de sorteios"),
            tooltip=[
                alt.Tooltip("inicio:Q", title="Faixa de p̂ (%): de", format=".1f"),
                alt.Tooltip("fim:Q", title="até", format=".1f"),
                alt.Tooltip("sorteios:Q", title="Sorteios"),
            ],
        )
    )
    camada_linhas = (
        alt.Chart(linhas)
        .mark_rule(strokeWidth=2.5)
        .encode(
            x=alt.X("valor:Q", scale=escala_x, title=eixo_x),
            color=alt.Color(
                "referencia:N",
                scale=alt.Scale(domain=nomes, range=[_ROXO, _VERDE, "#333333"][: len(nomes)]),
                legend=alt.Legend(title="Linhas", labelLimit=320, orient="bottom", direction="vertical"),
            ),
            strokeDash=alt.condition("datum.tracejado", alt.value([6, 4]), alt.value([1, 0])),
            tooltip=[
                alt.Tooltip("referencia:N", title="Linha"),
                alt.Tooltip("valor:Q", title="Valor (%)", format=".2f"),
            ],
        )
    )
    return alt.layer(camada_barras, camada_linhas).properties(
        title=alt.Title(text=titulo, subtitle=fonte, anchor="start"), width=_LARGURA, height=440
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
    eixo = _eixo(title=None, tickCount=5, labelOverlap="greedy")  # a unidade vai no rótulo da linha

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


# ---------------------------------------------------------------------------------------
# Região ou estado: as unidades, o ranking, o mapa, as três réguas e as condições
# ---------------------------------------------------------------------------------------

NIVEIS: tuple[str, ...] = ("regiao", "estado")
# Coluna dos dados que guarda a unidade em cada nível.
COLUNA_DO_NIVEL: dict[str, str] = {"regiao": "regiao", "estado": "uf"}


def _validar_nivel(nivel: str) -> None:
    if nivel not in NIVEIS:
        raise ValueError(f"nivel deve ser 'regiao' ou 'estado', e não {nivel!r}")


def unidades(ibge: pd.DataFrame, nivel: str) -> pd.DataFrame:
    """As unidades de um nível, com o nome por extenso, o grupo e as réguas do IBGE.

    `ibge` tem uma linha por UF com `uf, nome, regiao, pib_2023, populacao_2022`
    (`dados/ibge_uf.csv`). Devolve `codigo, nome, grupo, parte_pib, parte_populacao`:
    5 linhas no nível "regiao" (Norte, Nordeste, ...) e 27 no nível "estado"
    ("Pernambuco (PE)"). As partes somam 1 em cada nível.
    """
    _validar_nivel(nivel)
    base = ibge.copy()
    base["parte_pib"] = base["pib_2023"] / base["pib_2023"].sum()
    base["parte_populacao"] = base["populacao_2022"] / base["populacao_2022"].sum()
    if nivel == "estado":
        saida = pd.DataFrame(
            {
                "codigo": base["uf"],
                "nome": base["nome"] + " (" + base["uf"] + ")",
                "grupo": base["regiao"].map(GRUPO_DA_REGIAO),
                "parte_pib": base["parte_pib"],
                "parte_populacao": base["parte_populacao"],
            }
        )
    else:
        somas = base.groupby("regiao")[["parte_pib", "parte_populacao"]].sum().reindex(REGIOES)
        saida = pd.DataFrame(
            {
                "codigo": REGIOES,
                "nome": [NOME_REGIAO[r] for r in REGIOES],
                "grupo": [GRUPO_DA_REGIAO[r] for r in REGIOES],
                "parte_pib": somas["parte_pib"].to_numpy(),
                "parte_populacao": somas["parte_populacao"].to_numpy(),
            }
        )
    return saida.reset_index(drop=True)


def por_unidade(df: pd.DataFrame, unid: pd.DataFrame, nivel: str) -> pd.DataFrame:
    """Resumo dos contratos de `df` em cada unidade de `unid` (todas aparecem, mesmo sem contrato).

    Colunas acrescentadas: `contratos`, `participacao` (dos contratos), `valor`,
    `participacao_valor`, e, para spread e prazo, a média (`spread_media`,
    `prazo_amortizacao_media`) e o número de contratos com valor (`spread_n`,
    `prazo_amortizacao_n`). Unidade sem contrato fica com média ausente.
    """
    _validar_nivel(nivel)
    coluna = COLUNA_DO_NIVEL[nivel]
    grupos = df.groupby(coluna)
    saida = unid.copy()
    contratos = grupos.size().reindex(unid["codigo"], fill_value=0).to_numpy()
    valor = grupos["valor_contratado"].sum().reindex(unid["codigo"], fill_value=0.0).to_numpy()
    saida["contratos"] = contratos
    saida["participacao"] = contratos / contratos.sum() if contratos.sum() else 0.0
    saida["valor"] = valor
    saida["participacao_valor"] = valor / valor.sum() if valor.sum() else 0.0
    for variavel in ("spread", "prazo_amortizacao"):
        saida[f"{variavel}_media"] = grupos[variavel].mean().reindex(unid["codigo"]).to_numpy()
        saida[f"{variavel}_n"] = grupos[variavel].count().reindex(unid["codigo"], fill_value=0).to_numpy()
    return saida


def teto(valores, passo: float) -> float:
    """O menor múltiplo de `passo` acima do maior valor (o fim de um eixo que começa em zero)."""
    maior = float(np.nanmax(np.asarray(list(valores), dtype=float)))
    return float(passo * np.ceil(round(maior / passo, 9))) if maior > 0 else passo


def _eixo_unidades(nivel: str, ordem: list[str]) -> alt.Y:
    return alt.Y(
        "nome:N",
        sort=ordem,
        title=None,
        axis=alt.Axis(labelLimit=220, labelFontSize=12 if nivel == "regiao" else 11, ticks=False),
    )


# O Streamlit desenha os gráficos esticados com autosize "fit": a altura declarada inclui
# título, subtítulo, legenda e eixo. Cada altura soma essa sobra à área das barras.
_SOBRA = 120


def _altura(nivel: str, linhas_por_unidade: int = 1, por_linha: int | None = None, sobra: int = _SOBRA) -> int:
    """Altura que dá para ler o nome de cada unidade (5 regiões ou 27 estados), mais a sobra."""
    if por_linha is None:
        por_linha = 44 if nivel == "regiao" else 22
    return sobra + por_linha * linhas_por_unidade * (5 if nivel == "regiao" else 27)


def ranking(
    df: pd.DataFrame,
    unid: pd.DataFrame,
    nivel: str,
    titulo: str,
    *,
    max_x: float,
    medida: str = "contratos",
    legenda: tuple[str, ...] = (),
) -> alt.LayerChart:
    """Participação de cada unidade nos contratos (ou no valor), em barras horizontais da
    maior para a menor, na cor do grupo, com o valor escrito na ponta.

    O eixo vai de 0 a `max_x`; para os Painéis 1 e 2 ficarem na mesma escala, passe nos
    dois o mesmo `max_x` (calculado com a população e a amostra juntas).
    """
    if medida not in ("contratos", "valor"):
        raise ValueError(f"medida deve ser 'contratos' ou 'valor', e não {medida!r}")
    campo = "participacao" if medida == "contratos" else "participacao_valor"
    dados = por_unidade(df, unid, nivel)[["codigo", "nome", "grupo", "contratos", campo]]
    dados = dados.rename(columns={campo: "parte"})
    dados["rotulo"] = [_pct(v) for v in dados["parte"]]
    ordem = list(dados.sort_values("parte", ascending=False, kind="stable")["nome"])
    eixo = "% dos contratos" if medida == "contratos" else "% do valor contratado"
    y = _eixo_unidades(nivel, ordem)
    x = alt.X("parte:Q", scale=alt.Scale(domain=[0, max_x]), axis=alt.Axis(format=".0%", title=eixo, tickCount=6))
    tooltip = [
        alt.Tooltip("nome:N", title="Onde"),
        alt.Tooltip("grupo:N", title="Grupo"),
        alt.Tooltip("rotulo:N", title=eixo),
        alt.Tooltip("contratos:Q", title="Contratos", format=",d"),
    ]
    barras = (
        alt.Chart(dados)
        .mark_bar(cornerRadiusEnd=4, size=22 if nivel == "regiao" else 13)
        .encode(y=y, x=x, color=_cor_grupo(), tooltip=tooltip)
    )
    textos = (
        alt.Chart(dados)
        .mark_text(align="left", dx=4, fontSize=11, color="#333333")
        .encode(y=y, x=x, text="rotulo:N")
    )
    return alt.layer(barras, textos).properties(
        title=_titulo(titulo, *legenda), width=_LARGURA, height=_altura(nivel)
    )


def _rotulo_limiar(x: float) -> str:
    return f"{x * 100:.0f}%"


# Faixas do mapa, uma por cor da rampa: a legenda mostra as 7, sempre, nos dois painéis.
FAIXAS_MAPA = (
    [f"menos de {_rotulo_limiar(LIMIARES_MAPA[0])}"]
    + [f"{_rotulo_limiar(a)} a {_rotulo_limiar(b)}" for a, b in zip(LIMIARES_MAPA, LIMIARES_MAPA[1:])]
    + [f"{_rotulo_limiar(LIMIARES_MAPA[-1])} ou mais"]
)


def faixa_do_mapa(parte: float) -> str:
    """Faixa de `FAIXAS_MAPA` em que cai a participação (o limiar entra na faixa de cima)."""
    return FAIXAS_MAPA[int(np.searchsorted(LIMIARES_MAPA, parte, side="right"))]


def mapa(
    df: pd.DataFrame,
    unid: pd.DataFrame,
    nivel: str,
    malha: dict,
    titulo: str,
    *,
    legenda: tuple[str, ...] = (),
) -> alt.Chart:
    """Mapa do Brasil com a cor da unidade (região ou estado) pela % dos contratos.

    `malha` é o GeoJSON do nível (`dados/regiao.geojson` ou `dados/uf.geojson`), com a
    propriedade `regiao` ou `uf`. A escala de cor é sequencial, de um só matiz, em faixas
    com os limiares fixos de `LIMIARES_MAPA` (`FAIXAS_MAPA`): o mesmo domínio nos Painéis
    1 e 2 e nos dois níveis, e a legenda mostra todas as faixas.
    """
    resumo = por_unidade(df, unid, nivel).set_index("codigo")
    chave = "regiao" if nivel == "regiao" else "uf"
    feicoes = []
    for feicao in malha["features"]:
        codigo = feicao["properties"][chave]
        linha = resumo.loc[codigo]
        feicoes.append(
            {
                "type": "Feature",
                "geometry": feicao["geometry"],
                "properties": {
                    "nome": linha["nome"],
                    "grupo": NOME_GRUPO[linha["grupo"]],
                    "parte": float(linha["participacao"]),
                    "rotulo": _pct(linha["participacao"]),
                    "faixa": faixa_do_mapa(float(linha["participacao"])),
                    "contratos": int(linha["contratos"]),
                },
            }
        )
    return (
        alt.Chart(alt.Data(values=feicoes), title=_titulo(titulo, *legenda))
        .mark_geoshape(stroke="#ffffff", strokeWidth=0.8)
        .encode(
            color=alt.Color(
                "properties.faixa:O",
                scale=alt.Scale(domain=FAIXAS_MAPA, range=RAMPA_MAPA),
                legend=alt.Legend(title="% dos contratos", orient="right", symbolType="square",
                                  symbolSize=220, symbolStrokeWidth=0),
            ),
            tooltip=[
                alt.Tooltip("properties.nome:N", title="Onde"),
                alt.Tooltip("properties.grupo:N", title="Grupo"),
                alt.Tooltip("properties.rotulo:N", title="% dos contratos"),
                alt.Tooltip("properties.contratos:Q", title="Contratos", format=",d"),
            ],
        )
        .project(type="mercator")
        .properties(width=400, height=420)
    )


NOME_SOMA_NNE = "Norte e Nordeste (soma)"

REGUAS: dict[str, str] = {
    "contratos": "% dos contratos (BNDES)",
    "pib": "% do PIB (IBGE, 2023)",
    "populacao": "% da população (IBGE, 2022)",
}


def tres_reguas(
    df: pd.DataFrame,
    unid: pd.DataFrame,
    nivel: str,
    titulo: str,
    *,
    max_x: float,
    legenda: tuple[str, ...] = (),
    reguas_nne: tuple[float, float] | None = None,
) -> alt.LayerChart:
    """Para cada unidade, três barras lado a lado: % dos contratos de `df`, % do PIB e
    % da população. Ordenadas pela % dos contratos, da maior para a menor.

    Com `reguas_nne = (régua do PIB, régua da população)`, as réguas de H1 e H2, e no nível
    "regiao", a primeira linha é "Norte e Nordeste (soma)": os contratos do grupo somados,
    contra as duas réguas dos testes.

    A barra dos contratos leva a cor do grupo da unidade; PIB e população, dois cinzas
    (a régua é a referência, não o assunto). O eixo vai de 0 a `max_x`, o mesmo nos
    Painéis 1 e 2. No nível "regiao" cada barra traz o valor escrito; no "estado", com 81
    barras, o valor fica na dica e na tabela.
    """
    resumo = por_unidade(df, unid, nivel)
    ordem = list(resumo.sort_values("participacao", ascending=False, kind="stable")["nome"])
    serie_contratos = {"N+NE": "Contratos: Norte e Nordeste", "Demais": "Contratos: demais regiões"}
    linhas = []
    for _, u in resumo.iterrows():
        for regua, campo in (("contratos", "participacao"), ("pib", "parte_pib"), ("populacao", "parte_populacao")):
            linhas.append(
                {
                    "nome": u["nome"],
                    "regua": REGUAS[regua],
                    "serie": serie_contratos[u["grupo"]] if regua == "contratos" else REGUAS[regua],
                    "parte": float(u[campo]),
                    "rotulo": _pct(u[campo]),
                }
            )
    if reguas_nne is not None and nivel == "regiao":
        soma_nne = float(resumo.loc[resumo["grupo"] == "N+NE", "participacao"].sum())
        nome_soma = NOME_SOMA_NNE
        ordem = [nome_soma, *ordem]
        for regua, valor in zip(REGUAS, (soma_nne, *reguas_nne)):
            linhas.append({
                "nome": nome_soma,
                "regua": REGUAS[regua],
                "serie": serie_contratos["N+NE"] if regua == "contratos" else REGUAS[regua],
                "parte": float(valor),
                "rotulo": _pct(valor),
            })
    dados = pd.DataFrame(linhas)
    series = [serie_contratos["N+NE"], serie_contratos["Demais"], REGUAS["pib"], REGUAS["populacao"]]
    cor = alt.Color(
        "serie:N",
        scale=alt.Scale(domain=series, range=[CORES["N+NE"], CORES["Demais"], _CINZA_PIB, _CINZA_POP]),
        legend=alt.Legend(title="Régua", orient="top", direction="horizontal", labelLimit=260, columns=2),
    )
    y = _eixo_unidades(nivel, ordem)
    deslocamento = alt.YOffset("regua:N", sort=list(REGUAS.values()))
    x = alt.X("parte:Q", scale=alt.Scale(domain=[0, max_x]),
              axis=alt.Axis(format=".0%", title="Participação (%)", tickCount=10))
    barras = (
        alt.Chart(dados)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=y,
            yOffset=deslocamento,
            x=x,
            color=cor,
            tooltip=[
                alt.Tooltip("nome:N", title="Onde"),
                alt.Tooltip("regua:N", title="Régua"),
                alt.Tooltip("rotulo:N", title="Participação"),
            ],
        )
    )
    camadas = [barras]
    if nivel == "regiao":
        camadas.append(
            alt.Chart(dados)
            .mark_text(align="left", dx=4, fontSize=11, color="#333333")
            .encode(y=y, yOffset=deslocamento, x=x, text="rotulo:N")
        )
    fonte = [*legenda, FONTE, FONTE_IBGE]
    return alt.layer(*camadas).properties(
        title=alt.Title(text=titulo, subtitle=fonte, anchor="start"),
        width=_LARGURA,
        height=210 + (20 if nivel == "regiao" else 10) * 3 * len(ordem),
    )


# Média com menos de N_MINIMO_MEDIA contratos sai em barra clara: oscila muito de amostra para amostra.
N_MINIMO_MEDIA = 10
CONFIANCA_MEDIA = (f"{N_MINIMO_MEDIA} ou mais", f"menos de {N_MINIMO_MEDIA}: média instável")


def condicoes(
    df: pd.DataFrame,
    unid: pd.DataFrame,
    nivel: str,
    coluna: str,
    titulo: str,
    *,
    max_x: float,
    legenda: tuple[str, ...] = (),
) -> alt.LayerChart:
    """Média de `coluna` (spread ou prazo) em cada unidade, em barras horizontais da maior
    para a menor, na cor do grupo, com a média e o n escritos na ponta.

    Unidade sem contrato com valor aparece no fim, sem barra, com "sem contratos". Média
    com menos de `N_MINIMO_MEDIA` contratos sai em barra clara, com legenda. O eixo vai de
    0 a `max_x`, o mesmo nos Painéis 1 e 2.
    """
    if coluna not in ("spread", "prazo_amortizacao"):
        raise ValueError(f"coluna deve ser 'spread' ou 'prazo_amortizacao', e não {coluna!r}")
    resumo = por_unidade(df, unid, nivel)
    casas = 2 if coluna == "spread" else 1
    dados = pd.DataFrame(
        {
            "nome": resumo["nome"],
            "grupo": resumo["grupo"],
            "media": resumo[f"{coluna}_media"],
            "n": resumo[f"{coluna}_n"],
        }
    )
    dados["rotulo"] = [
        f"{_br(m, casas)} (n = {_br(n, 0)})" if n > 0 else "sem contratos" for m, n in zip(dados["media"], dados["n"])
    ]
    dados["posicao"] = dados["media"].fillna(0.0)
    dados["amostra"] = [CONFIANCA_MEDIA[0] if n >= N_MINIMO_MEDIA else CONFIANCA_MEDIA[1] for n in dados["n"]]
    ordem = list(dados.sort_values("posicao", ascending=False, kind="stable")["nome"])
    y = _eixo_unidades(nivel, ordem)
    eixo = ROTULOS[coluna].replace("Prazo de amortização", "Prazo médio").replace("Spread", "Spread médio")
    x = alt.X("posicao:Q", scale=alt.Scale(domain=[0, max_x]), axis=_eixo(title=eixo, tickCount=6))
    barras = (
        alt.Chart(dados)
        .mark_bar(cornerRadiusEnd=4, size=22 if nivel == "regiao" else 13)
        .encode(
            y=y,
            x=x,
            color=_cor_grupo(),
            # Domínio fixo: a mesma legenda nos dois painéis e nos dois níveis.
            opacity=alt.Opacity(
                "amostra:N",
                scale=alt.Scale(domain=list(CONFIANCA_MEDIA), range=[1.0, 0.35]),
                legend=alt.Legend(title="Contratos na média", labelLimit=260, symbolType="square",
                                  symbolFillColor="#5f6b7a", symbolStrokeWidth=0),
            ),
            tooltip=[
                alt.Tooltip("nome:N", title="Onde"),
                alt.Tooltip("grupo:N", title="Grupo"),
                alt.Tooltip("rotulo:N", title="Média (n)"),
            ],
        )
    )
    textos = (
        alt.Chart(dados)
        .mark_text(align="left", dx=4, fontSize=11, color="#333333")
        .encode(y=y, x=x, text="rotulo:N")
    )
    return alt.layer(barras, textos).properties(
        title=_titulo(titulo, *legenda), width=_LARGURA, height=_altura(nivel)
    )
