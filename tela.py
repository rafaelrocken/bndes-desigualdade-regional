"""Peças de tela compartilhadas pelas páginas: formatação brasileira e o painel descritivo.

O painel descritivo é uma função só, chamada pela página da população (Painel 1) e
pela da amostra (Painel 2). Assim os dois painéis têm os mesmos gráficos, na mesma
ordem e nas mesmas escalas; o da amostra acrescenta a coluna "População" ao lado de
cada medida.
"""
import numpy as np
import pandas as pd
import streamlit as st

import calculo as c
import graficos as g

GRUPOS = ["N+NE", "Demais"]
REGIOES = ["N", "NE", "CO", "SE", "S"]
_SOBRESCRITO = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

# Rótulos das medidas de `calculo.descritiva`, na ordem em que aparecem nas tabelas.
MEDIDAS = {
    "n": "n (contratos)",
    "media": "Média",
    "mediana": "Mediana",
    "desvio_padrao": "Desvio padrão",
    "q1": "1º quartil (Q1)",
    "q3": "3º quartil (Q3)",
    "minimo": "Mínimo",
    "maximo": "Máximo",
}


# --- Formatação brasileira -----------------------------------------------------------

def br(x: float, casas: int = 2) -> str:
    """Número com vírgula decimal, ponto de milhar e sinal de menos tipográfico."""
    texto = f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "−" + texto[1:] if texto.startswith("-") else texto


def inteiro(x: float) -> str:
    """Contagem: sem casas decimais, com ponto de milhar."""
    return br(float(x), 0)


def pct(x: float, casas: int = 1) -> str:
    """Proporção (0 a 1) escrita como porcentagem."""
    return br(100 * x, casas) + "%"


def fp(p: float) -> str:
    """Valor-p: quatro casas, ou notação científica (3,7 × 10⁻²³) abaixo de 0,001."""
    if p < 0.001:
        mantissa, expoente = f"{p:.1e}".split("e")
        return f"{mantissa.replace('.', ',')} × 10{str(int(expoente)).translate(_SOBRESCRITO)}"
    return br(p, 4)


def faixa(ic, formatar) -> str:
    """Intervalo escrito como [inferior; superior]."""
    return f"[{formatar(ic[0])}; {formatar(ic[1])}]"


def tabela(df: pd.DataFrame) -> None:
    """Mostra uma tabela já formatada, sem o índice numérico do pandas."""
    st.dataframe(df, hide_index=True, width="stretch")


def tabela_estatica(df: pd.DataFrame) -> None:
    """Tabela estática (`st.table`): quebra o texto longo em várias linhas em vez de cortar a célula."""
    st.table(df, hide_index=True)


def grafico(chart) -> None:
    """Mostra um gráfico Altair na largura da coluna (gráficos em facetas mantêm a largura própria)."""
    st.altair_chart(chart, width="stretch")


# --- Tabelas do painel descritivo ----------------------------------------------------

def _com_log_valor(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `valor_log10`, o logaritmo de base 10 do valor contratado em reais."""
    saida = df.copy()
    saida["valor_log10"] = np.log10(saida["valor_contratado"])
    return saida


def _descritiva_grupos(df: pd.DataFrame, coluna: str) -> dict[str, pd.Series]:
    """Medidas de `calculo.descritiva` para todos os contratos e para cada grupo."""
    saida = {"Todos": c.descritiva(df[coluna])}
    for grupo in GRUPOS:
        saida[grupo] = c.descritiva(df.loc[df["grupo"] == grupo, coluna])
    return saida


def tabela_descritiva(pop: pd.DataFrame, amostra: pd.DataFrame | None, coluna: str, casas: int = 2,
                      escala: float = 1.0) -> pd.DataFrame:
    """Uma linha por medida; uma coluna por grupo. Com `amostra`, cada grupo ganha o par
    "Amostra" e "População", lado a lado. `escala` divide os valores (ex.: reais em milhões).
    """
    blocos = {"População": _descritiva_grupos(pop, coluna)}
    if amostra is not None:
        blocos = {"Amostra": _descritiva_grupos(amostra, coluna), **blocos}
    linhas = []
    for chave, rotulo in MEDIDAS.items():
        linha = {"Medida": rotulo}
        for grupo in ["Todos", *GRUPOS]:
            for nome, medidas in blocos.items():
                valor = medidas[grupo][chave]
                cabecalho = grupo if amostra is None else f"{grupo}: {nome}"
                linha[cabecalho] = inteiro(valor) if chave == "n" else br(valor / escala, casas)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _frequencias(df: pd.DataFrame, coluna: str, ordem: list[str]) -> tuple[pd.Series, pd.Series]:
    contagem = df[coluna].value_counts().reindex(ordem, fill_value=0)
    return contagem, contagem / contagem.sum()


def tabela_regioes(pop: pd.DataFrame, amostra: pd.DataFrame | None) -> pd.DataFrame:
    """Contratos e valor por região: frequência, proporção dos contratos e proporção do valor."""
    ordem = list(pop["regiao"].value_counts().index)
    bases = {"População": pop} if amostra is None else {"Amostra": amostra, "População": pop}
    saida = pd.DataFrame({"Região": ordem, "Grupo": [g.GRUPO_DA_REGIAO[r] for r in ordem]})
    for nome, df in bases.items():
        contagem, proporcao = _frequencias(df, "regiao", ordem)
        valor = df.groupby("regiao")["valor_contratado"].sum().reindex(ordem, fill_value=0.0)
        prefixo = "" if amostra is None else f"{nome}: "
        saida[f"{prefixo}contratos"] = [inteiro(v) for v in contagem]
        saida[f"{prefixo}% dos contratos"] = [pct(v, 2) for v in proporcao]
        saida[f"{prefixo}% do valor"] = [pct(v, 2) for v in valor / valor.sum()]
    return saida


def tabela_grupos(pop: pd.DataFrame, amostra: pd.DataFrame | None) -> pd.DataFrame:
    """Contratos por grupo. Na amostra, a proporção é o p̂; na população, o π verdadeiro."""
    saida = pd.DataFrame({"Grupo": GRUPOS})
    if amostra is not None:
        contagem, proporcao = _frequencias(amostra, "grupo", GRUPOS)
        saida["Amostra: contratos"] = [inteiro(v) for v in contagem]
        saida["Amostra: proporção (p̂)"] = [pct(v, 2) for v in proporcao]
    contagem, proporcao = _frequencias(pop, "grupo", GRUPOS)
    prefixo = "" if amostra is None else "População: "
    saida[f"{prefixo}contratos"] = [inteiro(v) for v in contagem]
    saida[f"{prefixo}proporção (π)"] = [pct(v, 2) for v in proporcao]
    return saida


def tabela_qualitativa_por_grupo(pop: pd.DataFrame, amostra: pd.DataFrame | None, coluna: str,
                                 rotulo: str) -> pd.DataFrame:
    """Frequência e proporção de cada categoria dentro de cada grupo (soma 100% por grupo)."""
    ordem = list(pop[coluna].value_counts().index)
    saida = pd.DataFrame({rotulo: ordem})
    bases = {"População": pop} if amostra is None else {"Amostra": amostra, "População": pop}
    for grupo in GRUPOS:
        for nome, df in bases.items():
            contagem, proporcao = _frequencias(df[df["grupo"] == grupo], coluna, ordem)
            prefixo = grupo if amostra is None else f"{grupo}: {nome}"
            saida[f"{prefixo} (contratos)"] = [inteiro(v) for v in contagem]
            saida[f"{prefixo} (%)"] = [pct(v, 1) for v in proporcao]
    return saida


def tabela_qualidade(pop: pd.DataFrame, amostra: pd.DataFrame | None) -> pd.DataFrame:
    """Qualidade dos dados por grupo: sem spread em reais, spread = 0 e prazo = 0."""
    def contar(df: pd.DataFrame) -> dict[str, int]:
        return {
            "Contratos": len(df),
            "Sem spread em reais (fora de H3 e H4)": int(df["spread"].isna().sum()),
            "Spread = 0": int((df["spread"] == 0).sum()),
            "Prazo de amortização = 0": int((df["prazo_amortizacao"] == 0).sum()),
        }

    bases = {"População": pop} if amostra is None else {"Amostra": amostra, "População": pop}
    linhas = []
    for item in contar(pop):
        linha = {"Item": item}
        for grupo in GRUPOS:
            for nome, df in bases.items():
                cabecalho = grupo if amostra is None else f"{grupo}: {nome}"
                linha[cabecalho] = inteiro(contar(df[df["grupo"] == grupo])[item])
        linhas.append(linha)
    return pd.DataFrame(linhas)


# --- O painel descritivo -------------------------------------------------------------

def painel_descritivo(pop: pd.DataFrame, amostra: pd.DataFrame, qual: str) -> None:
    """Desenha o Painel 1 (`qual="populacao"`) ou o Painel 2 (`qual="amostra"`).

    Os gráficos são os mesmos nos dois, com o domínio dos eixos calculado na
    população e o eixo vertical dos histogramas fixado pelo maior valor entre
    população e amostra. As tabelas do Painel 2 trazem a coluna "População".
    """
    if qual not in ("populacao", "amostra"):
        raise ValueError(f'qual deve ser "populacao" ou "amostra", não {qual!r}')
    eh_amostra = qual == "amostra"
    pop, amostra = _com_log_valor(pop), _com_log_valor(amostra)
    df = amostra if eh_amostra else pop
    comparar = amostra if eh_amostra else None
    nome = f"amostra, n = {inteiro(len(amostra))}" if eh_amostra else f"população, N = {inteiro(len(pop))}"

    # 1. Região: contratos e valor, lado a lado
    st.subheader("Para onde vão os contratos")
    esquerda, direita = st.columns(2)
    with esquerda:
        grafico(g.barras_regiao(df, "contratos", f"Participação de cada região nos contratos ({nome})"))
    with direita:
        grafico(g.barras_regiao(df, "valor", f"Participação de cada região no valor contratado ({nome})"))
    st.caption(
        r"Os testes contam contratos; o BNDES e o IBGE costumam falar em valor. Um contrato de R\$ 1 mil "
        r"pesa como um de R\$ 10 bilhões na contagem, e por isso as duas leituras aparecem lado a lado."
    )
    tabela(tabela_grupos(pop, comparar))
    tabela(tabela_regioes(pop, comparar))

    # 2. Setor e porte dentro de cada grupo
    st.subheader("Composição por setor dentro de cada grupo")
    grafico(g.barras_setor(df, f"Setor dos contratos dentro de cada grupo ({nome})"))
    tabela(tabela_qualitativa_por_grupo(pop, comparar, "setor", "Setor"))
    st.markdown("**Porte do cliente dentro de cada grupo**")
    tabela(tabela_qualitativa_por_grupo(pop, comparar, "porte", "Porte"))

    # 3 e 4. Spread e prazo: histograma, boxplot e medidas descritivas
    variaveis = [
        ("spread", "Spread", "% a.a.", "Só os contratos com subcréditos indexados em reais; os demais ficam fora (tabela de qualidade, abaixo)."),
        ("prazo_amortizacao", "Prazo de amortização", "meses", "Todos os contratos."),
    ]
    for coluna, titulo, unidade, nota in variaveis:
        st.subheader(f"{titulo} ({unidade}) por grupo")
        dom = g.dominio(pop, coluna)
        max_y = g.max_proporcao_histograma([pop, amostra], coluna, dom)
        esquerda, direita = st.columns(2)
        with esquerda:
            grafico(g.histograma(df, coluna, dom, f"Distribuição do {titulo.lower()} ({nome})", max_y=max_y))
        with direita:
            grafico(g.boxplot(df, coluna, dom, f"{titulo} por grupo ({nome})"))
        st.caption(nota + " Mesmas escalas nos Painéis 1 e 2.")
        tabela(tabela_descritiva(pop, comparar, coluna))

    # 5. Valor contratado, em escala logarítmica
    st.subheader("Valor contratado (escala logarítmica)")
    rotulo_log = "Valor contratado (log10 de R$: 6 = R$ 1 milhão, 9 = R$ 1 bilhão)"
    dom = g.dominio(pop, "valor_log10")
    max_y = g.max_proporcao_histograma([pop, amostra], "valor_log10", dom)
    grafico(g.histograma(df, "valor_log10", dom, f"Distribuição do valor contratado ({nome})",
                         max_y=max_y, rotulo=rotulo_log))
    st.caption(
        "O valor vai de alguns milhares a bilhões de reais; na escala logarítmica cada passo do eixo "
        r"multiplica o valor por 10. A tabela abaixo está em R\$ milhões."
    )
    tabela(tabela_descritiva(pop, comparar, "valor_contratado", casas=1, escala=1e6))

    # 6. Qualidade dos dados
    st.subheader("Qualidade dos dados por grupo")
    tabela(tabela_qualidade(pop, comparar))
    st.caption(
        "Contratos sem nenhum subcrédito indexado em reais ficam fora do spread (H3, H4 e a descritiva "
        "do spread), mas continuam em H1, H2 e H5. Spreads e prazos iguais a 0, que parecem erro de "
        "registro, foram mantidos (limitação 5 do Painel 3)."
    )
