"""Peças de tela compartilhadas pelas páginas: formatação brasileira, cartões, abas, o
seletor "Ver por: Região | Estado" e o painel descritivo.

O painel descritivo é uma função só, chamada pela página da população (Painel 1) e
pela da amostra (Painel 2). Assim os dois painéis têm os mesmos gráficos, na mesma
ordem e nas mesmas escalas; o da amostra acrescenta a coluna "População" ao lado de
cada medida. Toda escala que depende dos dados (o fim dos eixos, o eixo vertical dos
histogramas) é calculada aqui com a população e a amostra juntas, e o mapa usa
limiares fixos; por isso a mesma escala sai nos dois painéis por construção.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

import calculo as c
import graficos as g
from dados_app import carregar

GRUPOS = ["N+NE", "Demais"]
REGIOES = ["N", "NE", "CO", "SE", "S"]
NOME_GRUPO = g.NOME_GRUPO
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

# O seletor "Ver por": a escolha vale nos dois painéis e pode vir da URL (?ver=estado).
CHAVE_NIVEL = "ver"
CHAVE_COPIA = "_nivel"
ROTULO_NIVEL = {"regiao": "Região", "estado": "Estado"}

ASSETS = Path(__file__).resolve().parent / "assets"  # logos; origem de cada um em assets/FONTES.md

# Caminhos das páginas, relativos ao app.py (para `st.page_link`).
PAGINA_BASE = "paginas/base.py"
def paginas_paineis() -> list[tuple[str, str, str]]:
    """(caminho, rótulo, ícone) dos três painéis; N e n saem dos dados."""
    pop, amostra, _ = carregar()
    return [
        ("paginas/populacao.py", f"Painel 1 — População: os {inteiro(len(pop))} contratos", ":material/public:"),
        ("paginas/amostra.py", f"Painel 2 — Amostra: os {inteiro(len(amostra))} contratos sorteados",
         ":material/casino:"),
        ("paginas/inferencia.py", "Painel 3 — Inferência: os cinco testes e a comparação", ":material/functions:"),
    ]


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


def com_sinal(x: float, casas: int = 1) -> str:
    """Número com sinal explícito: +1,2 ou −1,2 (zero sai sem sinal)."""
    texto = br(x, casas)
    return texto if texto.startswith("−") or float(f"{x:.{casas}f}") == 0 else "+" + texto


def fp(p: float) -> str:
    """Valor-p: quatro casas, ou notação científica (3,7 × 10⁻²³) abaixo de 0,001."""
    if p < 0.001:
        mantissa, expoente = f"{p:.1e}".split("e")
        return f"{mantissa.replace('.', ',')} × 10{str(int(expoente)).translate(_SOBRESCRITO)}"
    return br(p, 4)


def faixa(ic, formatar) -> str:
    """Intervalo escrito como [inferior; superior]."""
    return f"[{formatar(ic[0])}; {formatar(ic[1])}]"


def nota_regua_populacao(ibge: pd.DataFrame, x: int, n: int, alfa: float = 0.05) -> str:
    """A nota que acompanha a régua de H2 (35,4%): a soma exata no IBGE arredonda para outro
    número, porque o pré-registro somou as partes já arredondadas. Tudo calculado: as partes,
    a soma exata e o z do teste com cada régua, para dizer se a decisão muda."""
    partes = ibge.groupby("regiao")["populacao_2022"].sum() / ibge["populacao_2022"].sum()
    exata = float(partes["N"] + partes["NE"])
    p0 = c.HIPOTESES[1]["p0_ou_delta0"]
    z_pre = c.teste_z_proporcao(x, n, p0, alfa)
    z_exata = c.teste_z_proporcao(x, n, exata, alfa)
    mesma = z_pre["rejeita"] == z_exata["rejeita"]
    return (
        f"Nota sobre a régua da população: a soma exata no IBGE é {pct(exata, 2)}, que arredonda para "
        f"{pct(exata, 1)}. O pré-registro fixou {pct(p0, 1)}, a soma das partes já arredondadas "
        f"(NE {pct(round(float(partes['NE']), 3))} + N {pct(round(float(partes['N']), 3))}). "
        f"Com a régua do pré-registro, z = {br(z_pre['z'], 2)}; com a soma exata, z = {br(z_exata['z'], 2)}: "
        + ("a decisão não muda." if mesma else "a decisão muda.")
    )


def extenso(texto: str) -> str:
    """Troca a sigla dos grupos pelo nome por extenso, para textos corridos."""
    return texto.replace("N+NE", "Norte e Nordeste")


# --- Peças de tela -------------------------------------------------------------------

def tabela(df: pd.DataFrame) -> None:
    """Mostra uma tabela já formatada, sem o índice numérico do pandas."""
    st.dataframe(df, hide_index=True, width="stretch")


def tabela_estatica(df: pd.DataFrame) -> None:
    """Tabela estática (`st.table`): quebra o texto longo em várias linhas em vez de cortar a célula."""
    st.table(df, hide_index=True)


def grafico(chart) -> None:
    """Mostra um gráfico Altair na largura da coluna (gráficos em facetas mantêm a largura própria)."""
    st.altair_chart(chart, width="stretch")


def numeros(*tabelas: pd.DataFrame, rotulo: str = "Ver os números", aberto: bool = False) -> None:
    """As tabelas de números, num expansor abaixo do gráfico: recolhido, salvo com `aberto=True`
    (as medidas descritivas e as frequências que o enunciado pede ficam visíveis sem clique)."""
    with st.expander(rotulo, expanded=aberto):
        for df in tabelas:
            tabela(df)


def pergunta(texto: str) -> None:
    """A pergunta que a página responde, em uma frase, logo abaixo do título."""
    st.markdown(f"#### {texto}")


def cartoes(itens: list[dict]) -> None:
    """Uma linha de números grandes (`st.metric`). Cada item: `rotulo`, `valor`, e
    opcionalmente `delta` (texto, já com o sinal) e `ajuda`.

    O delta sai em cinza e sem seta: aqui ele mede distância (amostra − população), e
    subir ou descer não é bom nem ruim.
    """
    for coluna, item in zip(st.columns(len(itens)), itens):
        with coluna:
            st.metric(
                item["rotulo"],
                item["valor"],
                delta=item.get("delta"),
                delta_color="off",
                delta_arrow="off",
                help=item.get("ajuda"),
                border=True,
            )


def abas(opcoes: dict[str, str]):
    """`st.tabs` com os rótulos de `opcoes` ({atalho: rótulo}). A aba inicial pode vir da URL,
    `?aba=<atalho>`, o que serve para abrir uma aba direto no vídeo e nas fotos de conferência.
    """
    pedida = st.query_params.get("aba")
    padrao = opcoes.get(pedida) if pedida else None
    return st.tabs(list(opcoes.values()), default=padrao)


def link_pagina(caminho: str, rotulo: str, icone: str | None = None) -> None:
    """Link para outra página do app (`st.page_link`).

    Quando a página roda sozinha, sem o `st.navigation` do `app.py` (testes, ou
    `streamlit run paginas/x.py`), as outras páginas não existem e o link vira texto.
    """
    try:
        st.page_link(caminho, label=rotulo, icon=icone)
    except StreamlitAPIException:
        st.caption(rotulo)


def bloco(titulo: str | None = None):
    """Caixa com borda que agrupa uma pergunta com o seu gráfico e os seus números.
    Uso: `with bloco("Título"):`."""
    caixa = st.container(border=True)
    if titulo:
        caixa.subheader(titulo)
    return caixa


# --- Cabeçalho e rodapé, iguais em todas as páginas ----------------------------------

TITULO_APP = "O BNDES compensa a desigualdade regional?"
AUTORIA = (
    "Especialização em Inteligência Artificial Avançada · UFRGS · "
    "IA003 Probabilidade e Estatística para IA · Rafael Rockenbach · 2026"
)
AVISO_FONTES = (
    "Trabalho acadêmico independente, feito com dados abertos. "
    "Sem vínculo com o BNDES nem com o IBGE, e sem endosso deles."
)
# Os logos do BNDES e do IBGE ficam só no rodapé de fontes: o app não é uma publicação deles.
FONTES_IBGE = [
    ("PIB das Contas Regionais 2023 (SIDRA, tabela 5938)", "https://sidra.ibge.gov.br/tabela/5938"),
    ("População do Censo 2022 (SIDRA, tabela 4709)", "https://sidra.ibge.gov.br/tabela/4709"),
    ("Malhas territoriais (API de malhas v3)", "https://servicodados.ibge.gov.br/api/docs/malhas?versao=3"),
]


def _logo(nome: str) -> str:
    return str(ASSETS / nome)


def cabecalho() -> None:
    """Cabeçalho comum: o logo da UFRGS no canto (`st.logo`) e, numa faixa com borda, o
    logo da UFRGS, a pergunta do trabalho e a autoria."""
    st.logo(_logo("ufrgs.svg"), size="large", link="https://www.ufrgs.br/")
    with st.container(border=True):
        ufrgs, texto = st.columns([0.7, 9], vertical_alignment="center", gap="medium")
        ufrgs.image(_logo("ufrgs.svg"), width=64)
        texto.markdown(f"**{TITULO_APP}**")
        texto.caption(AUTORIA)


def rodape() -> None:
    """Rodapé de fontes: os logos do BNDES e do IBGE, pequenos, cada um com o link da base
    usada, e o aviso de que o trabalho é independente."""
    meta = carregar()[2]
    st.divider()
    st.markdown("**Fontes dos dados**")
    bndes, ibge = st.columns(2, gap="large")
    with bndes:
        st.image(_logo("bndes.svg"), width=96)
        st.markdown(f"[Operações de financiamento: operações não automáticas (Portal de Dados Abertos do BNDES)]"
                    f"({meta['fonte_url']})")
    with ibge:
        st.image(_logo("ibge.svg"), width=96)
        st.markdown("  \n".join(f"[{rotulo}]({url})" for rotulo, url in FONTES_IBGE))
    st.caption(AVISO_FONTES)


def voltar_para_base() -> None:
    """Linha discreta, no topo de cada painel, com o link de volta para a página da base."""
    link_pagina(PAGINA_BASE, "Base de dados e amostra", ":material/arrow_back:")


def seletor_nivel(pagina: str) -> str:
    """O seletor "Ver por: Região | Estado" da `pagina` ("populacao" ou "amostra").
    Devolve "regiao" ou "estado".

    A escolha comum aos dois painéis fica em `st.session_state["_nivel"]`, que não é
    chave de widget e por isso sobrevive à troca de página. Cada página tem o seu widget
    (chave `ver_<pagina>`), semeado com essa escolha antes de ser desenhado e que a
    atualiza quando o usuário clica. Na primeira abertura, a escolha vem da URL
    (`?ver=estado`), e a URL acompanha a escolha. O padrão é Região.
    """
    if st.session_state.get(CHAVE_COPIA) not in ROTULO_NIVEL:
        da_url = st.query_params.get(CHAVE_NIVEL)
        st.session_state[CHAVE_COPIA] = da_url if da_url in ROTULO_NIVEL else "regiao"
    chave = f"{CHAVE_NIVEL}_{pagina}"
    if st.session_state.get(chave) != st.session_state[CHAVE_COPIA]:
        st.session_state[chave] = st.session_state[CHAVE_COPIA]

    def guardar() -> None:
        if st.session_state.get(chave) in ROTULO_NIVEL:
            st.session_state[CHAVE_COPIA] = st.session_state[chave]

    st.segmented_control(
        "Ver por",
        options=list(ROTULO_NIVEL),
        format_func=ROTULO_NIVEL.get,
        required=True,
        key=chave,
        on_change=guardar,
    )
    escolha = st.session_state[CHAVE_COPIA]
    if st.query_params.get(CHAVE_NIVEL) != escolha:
        st.query_params[CHAVE_NIVEL] = escolha
    return escolha


# --- Tabelas do painel descritivo ----------------------------------------------------

def _com_log_valor(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `valor_log10`, o logaritmo de base 10 do valor contratado em reais."""
    saida = df.copy()
    saida["valor_log10"] = np.log10(saida["valor_contratado"])
    return saida


def _com_nomes(df: pd.DataFrame) -> pd.DataFrame:
    """Setor e porte por extenso e com acento, para gráficos e tabelas."""
    saida = df.copy()
    saida["setor"] = saida["setor"].map(g.NOME_SETOR).fillna(saida["setor"])
    saida["porte"] = saida["porte"].map(g.NOME_PORTE).fillna(saida["porte"])
    return saida


def _descritiva_grupos(df: pd.DataFrame, coluna: str) -> dict[str, pd.Series]:
    """Medidas de `calculo.descritiva` para todos os contratos e para cada grupo."""
    saida = {"Todos": c.descritiva(df[coluna])}
    for grupo in GRUPOS:
        saida[grupo] = c.descritiva(df.loc[df["grupo"] == grupo, coluna])
    return saida


def _nome(grupo: str) -> str:
    return NOME_GRUPO.get(grupo, grupo)


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
                cabecalho = _nome(grupo) if amostra is None else f"{_nome(grupo)}: {nome}"
                linha[cabecalho] = inteiro(valor) if chave == "n" else br(valor / escala, casas)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _frequencias(df: pd.DataFrame, coluna: str, ordem: list[str]) -> tuple[pd.Series, pd.Series]:
    contagem = df[coluna].value_counts().reindex(ordem, fill_value=0)
    return contagem, contagem / contagem.sum()


def tabela_grupos(pop: pd.DataFrame, amostra: pd.DataFrame | None) -> pd.DataFrame:
    """Contratos por grupo. Na amostra, a proporção é o p̂; na população, o π verdadeiro."""
    saida = pd.DataFrame({"Grupo": [_nome(gr) for gr in GRUPOS]})
    if amostra is not None:
        contagem, proporcao = _frequencias(amostra, "grupo", GRUPOS)
        saida["Amostra: contratos"] = [inteiro(v) for v in contagem]
        saida["Amostra: proporção (p̂)"] = [pct(v, 2) for v in proporcao]
    contagem, proporcao = _frequencias(pop, "grupo", GRUPOS)
    prefixo = "" if amostra is None else "População: "
    saida[f"{prefixo}contratos"] = [inteiro(v) for v in contagem]
    saida[f"{prefixo}proporção (π)"] = [pct(v, 2) for v in proporcao]
    return saida


def tabela_unidades(pop: pd.DataFrame, amostra: pd.DataFrame | None, unid: pd.DataFrame, nivel: str,
                    campos: list[tuple[str, str, object]], reguas: bool = False) -> pd.DataFrame:
    """Uma linha por região ou estado, na ordem da população (% dos contratos, da maior para
    a menor). `campos` é uma lista de (coluna de `graficos.por_unidade`, rótulo, formatador);
    com `amostra`, cada campo ganha o par "Amostra" e "População". Com `reguas`, acrescenta
    a % do PIB e a % da população (IBGE), que são as mesmas nos dois painéis.
    """
    resumo_pop = g.por_unidade(pop, unid, nivel)
    ordem = resumo_pop.sort_values("participacao", ascending=False, kind="stable").index
    bases = {"População": resumo_pop}
    if amostra is not None:
        bases = {"Amostra": g.por_unidade(amostra, unid, nivel), **bases}
    saida = pd.DataFrame({"Onde": unid["nome"], "Grupo": [_nome(gr) for gr in unid["grupo"]]})
    for coluna, rotulo, formatar in campos:
        for nome, resumo in bases.items():
            cabecalho = rotulo if amostra is None else f"{rotulo}: {nome}"
            saida[cabecalho] = [formatar(v) if pd.notna(v) else "sem contratos" for v in resumo[coluna]]
    if reguas:
        saida["% do PIB (IBGE 2023)"] = [pct(v, 1) for v in unid["parte_pib"]]
        saida["% da população (IBGE 2022)"] = [pct(v, 1) for v in unid["parte_populacao"]]
    return saida.loc[ordem].reset_index(drop=True)


def tabela_qualitativa_por_grupo(pop: pd.DataFrame, amostra: pd.DataFrame | None, coluna: str,
                                 rotulo: str) -> pd.DataFrame:
    """Frequência e proporção de cada categoria dentro de cada grupo (soma 100% por grupo)."""
    ordem = list(pop[coluna].value_counts().index)
    saida = pd.DataFrame({rotulo: ordem})
    bases = {"População": pop} if amostra is None else {"Amostra": amostra, "População": pop}
    for grupo in GRUPOS:
        for nome, df in bases.items():
            contagem, proporcao = _frequencias(df[df["grupo"] == grupo], coluna, ordem)
            prefixo = _nome(grupo) if amostra is None else f"{_nome(grupo)}: {nome}"
            saida[f"{prefixo} (contratos)"] = [inteiro(v) for v in contagem]
            saida[f"{prefixo} (%)"] = [pct(v, 1) for v in proporcao]
    return saida


def _contar_qualidade(df: pd.DataFrame) -> dict[str, int]:
    return {
        "Contratos": len(df),
        "Sem spread em reais (fora de H3 e H4)": int(df["spread"].isna().sum()),
        "Spread = 0": int((df["spread"] == 0).sum()),
        "Prazo de amortização = 0": int((df["prazo_amortizacao"] == 0).sum()),
    }


def tabela_qualidade(pop: pd.DataFrame, amostra: pd.DataFrame | None) -> pd.DataFrame:
    """Qualidade dos dados por grupo: sem spread em reais, spread = 0 e prazo = 0."""
    bases = {"População": pop} if amostra is None else {"Amostra": amostra, "População": pop}
    linhas = []
    for item in _contar_qualidade(pop):
        linha = {"Item": item}
        for grupo in GRUPOS:
            for nome, df in bases.items():
                cabecalho = _nome(grupo) if amostra is None else f"{_nome(grupo)}: {nome}"
                linha[cabecalho] = inteiro(_contar_qualidade(df[df["grupo"] == grupo])[item])
        linhas.append(linha)
    return pd.DataFrame(linhas)


# --- Os cartões dos Painéis 1 e 2 ----------------------------------------------------

def _indicadores(df: pd.DataFrame) -> dict[str, float]:
    nne = df["grupo"] == "N+NE"
    media = df.groupby("grupo")[["spread", "prazo_amortizacao"]].mean()
    return {
        "contratos": float(nne.mean()),
        "valor": float(df.loc[nne, "valor_contratado"].sum() / df["valor_contratado"].sum()),
        "spread": float(media.loc["N+NE", "spread"] - media.loc["Demais", "spread"]),
        "prazo": float(media.loc["N+NE", "prazo_amortizacao"] - media.loc["Demais", "prazo_amortizacao"]),
    }


def cartoes_descritivos(pop: pd.DataFrame, amostra: pd.DataFrame, qual: str) -> None:
    """Os quatro números grandes dos Painéis 1 e 2. No Painel 2, o valor é o da amostra e o
    delta é a diferença para a população."""
    ind_pop, ind_amo = _indicadores(pop), _indicadores(amostra)
    ind = ind_amo if qual == "amostra" else ind_pop

    def delta(chave: str, casas: int, unidade: str, escala: float = 1.0) -> str | None:
        if qual != "amostra":
            return None
        return f"{com_sinal(escala * (ind_amo[chave] - ind_pop[chave]), casas)} {unidade} da população"

    cartoes([
        {"rotulo": "Contratos em Norte e Nordeste", "valor": pct(ind["contratos"], 1),
         "delta": delta("contratos", 1, "p.p.", 100),
         "ajuda": "Proporção dos contratos cuja UF principal fica no Norte ou no Nordeste. É o π de H1 e H2."},
        {"rotulo": "Valor contratado em Norte e Nordeste", "valor": pct(ind["valor"], 1),
         "delta": delta("valor", 1, "p.p.", 100),
         "ajuda": "A mesma proporção, mas somando reais em vez de contar contratos."},
        {"rotulo": "Spread: Norte e Nordeste − demais", "valor": f"{com_sinal(ind['spread'], 2)} p.p.",
         "delta": delta("spread", 2, "p.p."),
         "ajuda": "Diferença entre os spreads médios (% ao ano). Negativo: Norte e Nordeste pagam menos. É o δ de H3 e H4."},
        {"rotulo": "Prazo: Norte e Nordeste − demais", "valor": f"{com_sinal(ind['prazo'], 1)} meses",
         "delta": delta("prazo", 1, "meses"),
         "ajuda": "Diferença entre os prazos médios de amortização. Positivo: Norte e Nordeste têm mais prazo. É o parâmetro de H5."},
    ])


# --- O painel descritivo -------------------------------------------------------------

def _leitura(texto: str) -> None:
    """O texto de leitura de um gráfico: uma ou duas frases, logo acima dele."""
    st.markdown(texto)


def _unidade_maior(resumo: pd.DataFrame, campo: str, grupo: str | None = None) -> pd.Series:
    base = resumo if grupo is None else resumo[resumo["grupo"] == grupo]
    return base.sort_values(campo, ascending=False, kind="stable").iloc[0]


def painel_descritivo(pop: pd.DataFrame, amostra: pd.DataFrame, qual: str, ibge: pd.DataFrame,
                      malhas: dict[str, dict], nivel: str) -> None:
    """Desenha o Painel 1 (`qual="populacao"`) ou o Painel 2 (`qual="amostra"`), no nível
    `nivel` ("regiao" ou "estado"), em três abas.

    Os gráficos são os mesmos nos dois painéis. O fim de cada eixo é calculado com a
    população e a amostra juntas (nos dois níveis), o domínio dos histogramas na
    população, e o mapa usa limiares fixos: a mesma escala nos dois painéis. As tabelas
    do Painel 2 trazem a coluna "População".
    """
    if qual not in ("populacao", "amostra"):
        raise ValueError(f'qual deve ser "populacao" ou "amostra", não {qual!r}')
    g._validar_nivel(nivel)
    eh_amostra = qual == "amostra"
    pop, amostra = _com_nomes(_com_log_valor(pop)), _com_nomes(_com_log_valor(amostra))
    df = amostra if eh_amostra else pop
    comparar = amostra if eh_amostra else None
    base = f"Amostra, n = {inteiro(len(amostra))}" if eh_amostra else f"População, N = {inteiro(len(pop))}"
    onde = "região" if nivel == "regiao" else "estado"
    por_onde = "por região" if nivel == "regiao" else "por estado"
    unid = g.unidades(ibge, nivel)
    resumo = g.por_unidade(df, unid, nivel)
    resumos = [g.por_unidade(d, unid, nivel) for d in (pop, amostra)]
    malha = malhas[nivel]

    # Fins de eixo comuns aos dois painéis, calculados com a população e a amostra juntas.
    # A folga de 15% deixa espaço para o valor escrito na ponta da barra maior.
    def fim(valores, passo: float) -> float:
        return g.teto([1.15 * v for v in valores], passo)

    max_contratos = fim([v for r in resumos for v in r["participacao"]], 0.05)
    max_valor = fim([v for r in resumos for v in r["participacao_valor"]], 0.05)
    # Inclui a linha "Norte e Nordeste (soma)" do modo Região: as duas réguas e a soma dos contratos.
    soma_nne = [float(r.loc[r["grupo"] == "N+NE", "participacao"].sum()) for r in resumos]
    reguas_testes = [h["p0_ou_delta0"] for h in c.HIPOTESES[:2]]
    max_reguas = fim([v for r in resumos for col in ("participacao", "parte_pib", "parte_populacao")
                      for v in r[col]] + soma_nne + reguas_testes, 0.05)
    max_spread = fim([v for r in resumos for v in r["spread_media"].dropna()], 0.5)
    max_prazo = fim([v for r in resumos for v in r["prazo_amortizacao_media"].dropna()], 20)

    aba_onde, aba_condicoes, aba_qualidade = abas({
        "onde": "Onde estão os contratos",
        "condicoes": "Condições: spread e prazo",
        "qualidade": "Qualidade dos dados",
    })

    # ---------------------------------------------------------------- Onde estão
    with aba_onde:
        with bloco():
            maior = _unidade_maior(resumo, "participacao")
            nne = resumo.loc[resumo["grupo"] == "N+NE", "participacao"].sum()
            n_nne = int((unid["grupo"] == "N+NE").sum())
            if nivel == "regiao":
                frase = (f"O **{maior['nome']}** concentra **{pct(maior['participacao'])} dos contratos**; "
                         f"Norte e Nordeste juntos ficam com **{pct(nne)}**.")
            else:
                comparacao = "mais" if maior["participacao"] > nne else "menos"
                frase = (f"Só **{maior['nome']}** tem **{pct(maior['participacao'])} dos contratos**, {comparacao} "
                         f"que os {n_nne} estados do Norte e do Nordeste juntos (**{pct(nne)}**).")
            _leitura(frase + " Quanto mais escuro no mapa, maior a parte dos contratos; o ranking ao lado dá o número.")
            esquerda, direita = st.columns([1, 1.15], gap="medium")
            with esquerda:
                grafico(g.mapa(df, unid, nivel, malha, f"% dos contratos {por_onde}", legenda=(base,)))
            with direita:
                grafico(g.ranking(df, unid, nivel, f"Ranking {por_onde}", max_x=max_contratos,
                                  legenda=(f"{base}. Cor: grupo de regiões.",)))
            numeros(
                tabela_grupos(pop, comparar),
                tabela_unidades(pop, comparar, unid, nivel, [
                    ("contratos", "contratos", inteiro),
                    ("participacao", "% dos contratos", lambda v: pct(v, 2)),
                    ("participacao_valor", "% do valor", lambda v: pct(v, 2)),
                ]),
                aberto=True,
            )

        with bloco("Três réguas: contratos, PIB e população"):
            destaque = _unidade_maior(resumo, "participacao", "N+NE")
            p0_pib, p0_pop = (h["p0_ou_delta0"] for h in c.HIPOTESES[:2])
            soma_pop_exata = float(unid.loc[unid["grupo"] == "N+NE", "parte_populacao"].sum())
            _leitura(
                f"**{destaque['nome']}: {pct(destaque['participacao'])} dos contratos, {pct(destaque['parte_pib'])} do PIB, "
                f"{pct(destaque['parte_populacao'])} da população.** Norte e Nordeste juntos têm {pct(nne)} dos contratos, "
                f"a {br(100 * abs(nne - p0_pib), 1)} ponto percentual da sua parte no PIB ({pct(p0_pib)}, régua de H1) "
                f"e a {br(100 * abs(nne - p0_pop), 1)} pontos da sua parte na população ({pct(p0_pop)}, régua de H2; "
                f"o pré-registro somou as partes já arredondadas, e a soma exata no IBGE dá {pct(soma_pop_exata, 2)})."
                + (f" A primeira linha do gráfico, {g.NOME_SOMA_NNE}, põe essa comparação lado a lado."
                   if nivel == "regiao" else "")
            )
            grafico(g.tres_reguas(df, unid, nivel, f"Contratos, PIB e população {por_onde}", max_x=max_reguas,
                                  legenda=(f"{base}. Ordem: % dos contratos, da maior para a menor.",),
                                  reguas_nne=(p0_pib, p0_pop)))
            numeros(tabela_unidades(pop, comparar, unid, nivel,
                                    [("participacao", "% dos contratos", lambda v: pct(v, 1))], reguas=True))

        with bloco("E em valor contratado?"):
            valor_nne = resumo.loc[resumo["grupo"] == "N+NE", "participacao_valor"].sum()
            maior_ou_menor = "maiores" if valor_nne > nne else "menores"
            _leitura(
                f"Somando reais em vez de contar contratos, Norte e Nordeste ficam com **{pct(valor_nne)} do valor**, "
                f"contra {pct(nne)} dos contratos: os contratos de lá são, em média, {maior_ou_menor}. Os testes contam "
                r"contratos; o BNDES e o IBGE costumam falar em valor, e um contrato de R\$ 1 mil pesa como um de "
                r"R\$ 10 bilhões na contagem."
            )
            esquerda, direita = st.columns(2, gap="medium")
            with esquerda:
                grafico(g.ranking(df, unid, nivel, f"% do valor contratado {por_onde}", max_x=max_valor, medida="valor",
                                  legenda=(base,)))
            with direita:
                rotulo_log = "Valor contratado (log10 de R$: 6 = R$ 1 milhão, 9 = R$ 1 bilhão)"
                dom = g.dominio(pop, "valor_log10")
                max_y = g.max_proporcao_histograma([pop, amostra], "valor_log10", dom)
                grafico(g.histograma(df, "valor_log10", dom, "Tamanho dos contratos", max_y=max_y, rotulo=rotulo_log,
                                     legenda=(f"{base}. Escala logarítmica: cada passo multiplica o valor por 10.",),
                                     largura=340))
            numeros(tabela_descritiva(pop, comparar, "valor_contratado", casas=1, escala=1e6),
                    rotulo=r"Ver os números (valor contratado em R\$ milhões)")

    # ---------------------------------------------------------------- Condições
    with aba_condicoes:
        with bloco():
            medias = df.groupby("grupo")[["spread", "prazo_amortizacao"]].mean()
            s_nne, s_dem = medias.loc["N+NE", "spread"], medias.loc["Demais", "spread"]
            p_nne, p_dem = medias.loc["N+NE", "prazo_amortizacao"], medias.loc["Demais", "prazo_amortizacao"]
            if abs(s_nne - s_dem) < 0.05 * s_dem:
                sobre_spread = f"O spread médio é **quase igual** nos dois grupos ({br(s_nne, 2)} contra {br(s_dem, 2)} % a.a.)"
            else:
                sobre_spread = (f"O spread médio é **{'menor' if s_nne < s_dem else 'maior'}** em Norte e Nordeste "
                                f"({br(s_nne, 2)} contra {br(s_dem, 2)} % a.a.)")
            sobre_prazo = (f"o prazo médio de amortização é **{'maior' if p_nne > p_dem else 'menor'}** em Norte e Nordeste "
                           f"({br(p_nne, 1)} contra {br(p_dem, 1)} meses)")
            _leitura(f"{sobre_spread}; {sobre_prazo}. Abaixo, a média de cada {onde}, com o número de contratos (n) "
                     "ao lado: com poucos contratos, a média oscila mais.")
            esquerda, direita = st.columns(2, gap="medium")
            with esquerda:
                grafico(g.condicoes(df, unid, nivel, "spread", f"Spread médio {por_onde}", max_x=max_spread,
                                    legenda=(f"{base}. Só contratos indexados em reais.",)))
            with direita:
                grafico(g.condicoes(df, unid, nivel, "prazo_amortizacao", f"Prazo médio {por_onde}", max_x=max_prazo,
                                    legenda=(base,)))
            numeros(tabela_unidades(pop, comparar, unid, nivel, [
                ("spread_media", "spread médio (% a.a.)", lambda v: br(v, 2)),
                ("spread_n", "n do spread", inteiro),
                ("prazo_amortizacao_media", "prazo médio (meses)", lambda v: br(v, 1)),
                ("prazo_amortizacao_n", "n do prazo", inteiro),
            ]))

        variaveis = [
            ("spread", "Spread", "% a.a.", 2,
             "Só os contratos com subcréditos indexados em reais; os demais ficam fora (aba Qualidade dos dados)."),
            ("prazo_amortizacao", "Prazo de amortização", "meses", 1, "Todos os contratos."),
        ]
        for coluna, titulo, unidade, casas, nota in variaveis:
            with bloco(f"{titulo} por grupo: a distribuição inteira"):
                mediana = df.groupby("grupo")[coluna].median()
                _leitura(
                    f"Metade dos contratos de Norte e Nordeste fica abaixo de **{br(mediana['N+NE'], casas)} {unidade}**, "
                    f"contra **{br(mediana['Demais'], casas)} {unidade}** nas demais regiões (a mediana, o traço branco "
                    "na caixa). O histograma mostra onde os contratos se concentram; a caixa vai do 1º ao 3º quartil."
                )
                dom = g.dominio(pop, coluna)
                max_y = g.max_proporcao_histograma([pop, amostra], coluna, dom)
                esquerda, direita = st.columns(2, gap="medium")
                with esquerda:
                    grafico(g.histograma(df, coluna, dom, f"{titulo}: histograma", max_y=max_y,
                                         legenda=(f"{base}. {nota}",), largura=340))
                with direita:
                    grafico(g.boxplot(df, coluna, dom, f"{titulo}: boxplot",
                                      legenda=(f"{base}. Pontos: contratos fora dos bigodes.",)))
                numeros(tabela_descritiva(pop, comparar, coluna), aberto=True)

        with bloco("Setor dos contratos em cada grupo"):
            infra = df.groupby("grupo")["setor"].apply(lambda s: (s == "Infraestrutura").mean())
            _leitura(
                f"Infraestrutura é **{pct(infra['N+NE'], 0)} dos contratos de Norte e Nordeste** e "
                f"{pct(infra['Demais'], 0)} nas demais regiões. Obras de infraestrutura têm prazos longos por natureza, "
                "então parte da diferença de prazo pode vir do setor, não da região (limitação 4 do Painel 3)."
            )
            grafico(g.barras_setor(df, "Setor dentro de cada grupo", legenda=(f"{base}. Cada grupo soma 100%.",)))
            numeros(tabela_qualitativa_por_grupo(pop, comparar, "setor", "Setor"),
                    tabela_qualitativa_por_grupo(pop, comparar, "porte", "Porte"),
                    rotulo="Ver os números (setor e porte do cliente)")

    # ---------------------------------------------------------------- Qualidade
    with aba_qualidade:
        with bloco():
            q = _contar_qualidade(df)
            sem = q["Sem spread em reais (fora de H3 e H4)"]
            _leitura(
                f"**{inteiro(sem)} contratos ({pct(sem / q['Contratos'])}) não têm subcrédito indexado em reais** e ficam "
                "fora do spread (H3, H4 e a descritiva do spread), mas continuam em H1, H2 e H5. Spreads e prazos iguais "
                "a 0, que parecem erro de registro, foram mantidos (limitação 5 do Painel 3)."
            )
            cartoes([
                {"rotulo": "Contratos", "valor": inteiro(q["Contratos"])},
                {"rotulo": "Sem spread em reais", "valor": inteiro(sem), "ajuda": "Ficam fora de H3 e H4."},
                {"rotulo": "Spread = 0", "valor": inteiro(q["Spread = 0"]), "ajuda": "Mantidos (limitação 5)."},
                {"rotulo": "Prazo de amortização = 0", "valor": inteiro(q["Prazo de amortização = 0"]),
                 "ajuda": "Mantidos (limitação 5)."},
            ])
            numeros(tabela_qualidade(pop, comparar), rotulo="Ver os números por grupo")
