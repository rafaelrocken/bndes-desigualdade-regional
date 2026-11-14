"""Peças de tela compartilhadas pelas páginas: formatação brasileira, cartões, abas, o
seletor "Ver por: Região | Estado" e o painel descritivo.

O painel descritivo é uma função só, chamada pela página da população (Painel 1) e
pela da amostra (Painel 2). Assim os dois painéis têm os mesmos gráficos, na mesma
ordem e nas mesmas escalas; o da amostra acrescenta a coluna "População" ao lado de
cada medida. Toda escala que depende dos dados (o fim dos eixos, o eixo vertical dos
histogramas) é calculada aqui com a população e a amostra juntas, e o mapa usa
limiares fixos; por isso a mesma escala sai nos dois painéis por construção.
"""
import re
from html import escape
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


def html_inline(texto: str) -> str:
    """Texto curto (com `**negrito**` do markdown) pronto para entrar num pedaço de HTML: escapa
    `<`, `>` e `&`, troca `**x**` por `<strong>x</strong>` e o cifrão por entidade, para o
    markdown do Streamlit não ler `$` como fórmula."""
    texto = escape(texto, quote=False).replace("\\$", "$").replace("$", "&#36;")
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto)


def _html(trecho: str, onde=None) -> None:
    """Desenha um pedaço de HTML (em `onde`, ou na página). Vai por `st.markdown` para o texto
    continuar legível pelos testes; o Streamlit descarta SVG, então tudo é HTML e CSS."""
    (onde or st).markdown(trecho, unsafe_allow_html=True)


# --- Cartões de indicador (no estilo do dashboard da IA001) ---------------------------
#
# Cor das notas: cinza (`neutro`) quando a diferença é pequena, para qualquer lado (até 1 p.p.
# nas proporções; até 5% da média das demais regiões no spread e no prazo); fora dessa faixa,
# verde (`bom`) quando o resultado favorece a promessa do BNDES ao Norte e Nordeste e
# vermelho (`crit`) quando a contraria. A seta (▲ acima, ▼ abaixo) só aparece nas notas
# coloridas. A faixa é simétrica para que um ruído de sorteio não pinte de verde o que os
# testes dizem que não difere.

LIMIAR_PROPORCAO = 0.01  # 1 ponto percentual, em proporção
# Spread e prazo usam a faixa relativa: 1 p.p. de spread seria um terço da média (~2,7% a.a.).
QUASE_IGUAL = 0.05  # "praticamente igual": menos de 5% da média das demais regiões (o critério da leitura)


def tom_nota(dif: float, sentido: int, igual: float = 0.0, limiar: float = LIMIAR_PROPORCAO) -> str:
    """"bom", "crit" ou "neutro" para uma diferença `dif` (Norte e Nordeste − referência).

    `sentido` = +1 quando mais favorece a promessa (proporção, prazo) e −1 quando menos favorece
    (spread). Até `igual` (em módulo) a diferença é "praticamente igual" e fica neutra."""
    if abs(dif) <= igual:
        return "neutro"
    if dif * sentido > 0:
        return "bom"
    return "crit" if abs(dif) > limiar else "neutro"


def nota_regua(valor: float, regua: float, nome: str) -> dict:
    """Nota de um cartão de proporção contra uma régua: "▼ 16,3 p.p. abaixo da régua da
    população (35,4%)"."""
    dif = valor - regua
    return {
        "texto": f"{br(100 * abs(dif), 1)} p.p. {'acima' if dif >= 0 else 'abaixo'} da régua {nome} ({pct(regua)})",
        "tom": tom_nota(dif, +1, igual=LIMIAR_PROPORCAO),
        "sobe": dif >= 0,
    }


def nota_condicao(dif: float, media_demais: float, sentido: int, promessa: str, limiar: float) -> dict:
    """Nota de um cartão de diferença de médias (spread, prazo) contra a promessa da política
    regional (`promessa`, ex.: "um spread menor"). Abaixo de 5% da média das demais regiões, a
    diferença é "praticamente igual"."""
    igual = QUASE_IGUAL * abs(media_demais)
    tom = tom_nota(dif, sentido, igual, limiar)
    lado = "maior" if dif > 0 else "menor"
    if abs(dif) <= igual:
        texto = f"praticamente igual; a promessa era {promessa}"
    elif tom == "bom":
        texto = f"{lado} em Norte e Nordeste, como promete a política regional"
    else:
        texto = f"{lado} em Norte e Nordeste; a promessa era {promessa}"
    return {"texto": texto, "tom": tom, "sobe": dif > 0}


def medidor_html(valor: float, marcas: list[tuple[float, str]], teto: float, cor: str | None = None) -> str:
    """Barra do valor (0 a `teto`) com um traço em cada referência de `marcas` [(valor, rótulo)],
    e o rótulo de cada traço embaixo. Só HTML e CSS."""
    def pos(x: float) -> str:
        return f"{100 * min(max(x / teto, 0), 1):.1f}%"

    descricao = f"{pct(valor)} frente a " + " e ".join(f"{pct(m)} ({r})" for m, r in marcas)
    estilo_cor = f";background:{cor}" if cor else ""
    tracos = "".join(f'<span class="marca" style="left:{pos(m)}" title="{escape(r)}: {pct(m)}"></span>'
                     for m, r in marcas)
    rotulos = "".join(f'<span style="left:{pos(m)}">{escape(r)}</span>' for m, r in marcas)
    return (f'<div class="medidor" role="img" aria-label="{escape(descricao)}">'
            f'<span class="cheio" style="width:{pos(valor)}{estilo_cor}"></span>{tracos}</div>'
            f'<div class="medidor-rotulos" aria-hidden="true">{rotulos}</div>')


def nota_html(nota: dict) -> str:
    seta = ""
    if nota["tom"] != "neutro":
        seta = f'<span aria-hidden="true">{"▲" if nota["sobe"] else "▼"}</span> '
    return f'<p class="k-nota {nota["tom"]}">{seta}{html_inline(nota["texto"])}</p>'


def cartao_html(item: dict) -> str:
    """Um cartão: rótulo (com a ajuda no `title`), número grande, linhas cinzas (`sub`),
    medidor opcional e notas coloridas."""
    ajuda = item.get("ajuda")
    icone = (f' <span class="k-ajuda" title="{escape(ajuda, quote=True)}" aria-label="{escape(ajuda, quote=True)}">?</span>'
             if ajuda else "")
    subs = item.get("sub") or []
    subs = [subs] if isinstance(subs, str) else subs
    partes = [
        f'<p class="k-rot">{html_inline(item["rotulo"])}{icone}</p>',
        f'<p class="k-val{" k-val-texto" if item.get("valor_texto") else ""}">{html_inline(item["valor"])}</p>',
        *(f'<p class="k-sub">{html_inline(s)}</p>' for s in subs),
    ]
    if item.get("medidor"):
        partes.append(medidor_html(**item["medidor"]))
    partes += [nota_html(n) for n in item.get("notas", [])]
    titulo = f' title="{escape(ajuda, quote=True)}"' if ajuda else ""
    return f'<div class="kpi"{titulo}>{"".join(partes)}</div>'


def cartoes_html(itens: list[dict]) -> str:
    return f'<div class="kpis" style="--colunas:{len(itens)}">{"".join(cartao_html(i) for i in itens)}</div>'


def cartoes(itens: list[dict]) -> None:
    """Uma linha de cartões de indicador. Cada item: `rotulo`, `valor` e, opcionais, `sub`
    (linha ou lista de linhas cinzas), `medidor` (argumentos de `medidor_html`), `notas`
    (de `nota_regua`/`nota_condicao`), `ajuda` (o texto de ajuda, no `title`)."""
    _html(cartoes_html(itens))


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


def bloco(titulo: str | None = None, subtitulo: str | None = None, secao: str | None = None):
    """Caixa com borda que agrupa uma pergunta com o seu gráfico e os seus números.

    No estilo da IA001: `titulo` é a conclusão do bloco, em negrito forte; `subtitulo`, a
    leitura curta em cinza; `secao`, o nome do bloco numa linha pequena acima.
    Uso: `with bloco("Conclusão", "Como ler o gráfico", secao="Nome"):`."""
    caixa = st.container(border=True)
    partes = []
    if secao:
        partes.append(f'<p class="e-secao">{html_inline(secao)}</p>')
    if titulo:
        partes.append(f'<p class="e-tit" role="heading" aria-level="3">{html_inline(titulo.replace("**", ""))}</p>')
    if subtitulo:
        partes.append(f'<p class="e-sub">{html_inline(subtitulo)}</p>')
    if partes:
        _html(f'<div class="e-painel">{"".join(partes)}</div>', caixa)
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


# O estilo da tela, comum a todas as páginas e injetado por `cabecalho()`. Vem do dashboard da
# IA001: a fonte Archivo (Google Fonts, por CSS: nenhuma dependência nova), títulos fortes e
# condensados, cartões com número grande, medidor e nota colorida. As cores são as da IA001,
# com o azul trocado pelo azul-ardósia do tema (`primaryColor`), porque o azul aqui é a cor do
# grupo "Demais regiões" nos gráficos. Os seletores levam `.stApp` na frente para vencer os
# estilos do próprio Streamlit.
CSS = r"""<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');
:root{--tinta:#0f2238;--tinta2:#42526a;--apagado:#6b7686;--fio:#dfe4ea;--ardosia:#3B4B63;
  --nne:#D55E00;--trilho:#e3e7ec;--cartao:#f3f5f8;--bom:#0b6b0b;--crit:#b3261e}
.stApp, .stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stApp summary, button[data-baseweb="tab"] p{
  font-family:"Archivo",system-ui,"Segoe UI",sans-serif}
.stApp{color:var(--tinta)}
[data-testid="stMainBlockContainer"]{padding-top:4.2rem}
.stApp h2, .stApp h3{font-weight:750;font-stretch:84%;letter-spacing:0}
.stTabs [data-baseweb="tab-list"]{gap:22px}
button[data-baseweb="tab"] p{font-size:1rem;font-weight:600}
/* cabeçalho de página */
.stApp .e-topo{margin:0 0 .2rem}
.stApp .e-sobre{font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--apagado);margin:0 0 .3rem}
.stApp .e-titulo{font-size:2rem;font-weight:800;font-stretch:80%;line-height:1.08;color:var(--tinta);margin:0}
.stApp .e-contexto{font-size:.98rem;color:var(--tinta2);margin:.5em 0 0;line-height:1.45}
.stApp .e-contexto strong{color:var(--tinta)}
.stApp .e-autoria{font-size:.78rem;color:var(--apagado);margin:.55em 0 0}
/* cartões de indicador */
.stApp .kpis{display:grid;grid-template-columns:repeat(var(--colunas,4),minmax(0,1fr));gap:14px;margin:6px 0 10px}
.stApp .kpi{background:var(--cartao);border-radius:10px;padding:14px 16px 12px}
.stApp .k-rot{font-size:.86rem;color:var(--tinta2);margin:0;font-weight:550;line-height:1.3}
.stApp .k-ajuda{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;
  border:1px solid var(--apagado);border-radius:50%;font-size:.66rem;color:var(--apagado);cursor:help;
  vertical-align:1px;margin-left:3px}
.stApp .k-val{font-size:2.15rem;font-weight:800;font-stretch:78%;margin:.08em 0 .05em;line-height:1.05;color:var(--tinta)}
.stApp .k-val.k-val-texto{font-size:1.35rem;margin:.2em 0 .15em}
.stApp .k-sub{font-size:.84rem;color:var(--tinta2);margin:.15em 0 0;line-height:1.3}
.stApp .k-nota{font-size:.84rem;color:var(--tinta2);margin:.4em 0 0;line-height:1.3}
.stApp .k-nota.bom{color:var(--bom);font-weight:600}
.stApp .k-nota.crit{color:var(--crit);font-weight:600}
.stApp .medidor{position:relative;height:8px;border-radius:99px;background:var(--trilho);margin-top:9px}
.stApp .medidor .cheio{position:absolute;left:0;top:0;bottom:0;border-radius:99px;background:var(--nne)}
.stApp .medidor .marca{position:absolute;top:-5px;bottom:-5px;width:2.5px;margin-left:-1px;
  background:var(--tinta);border-radius:2px}
.stApp .medidor-rotulos{position:relative;height:1.05em;font-size:.7rem;color:var(--apagado);margin-top:4px}
.stApp .medidor-rotulos span{position:absolute;transform:translateX(-50%);white-space:nowrap}
@media (max-width:900px){.stApp .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){.stApp .kpis{grid-template-columns:minmax(0,1fr)}}
/* título de bloco que diz a conclusão */
.stApp .e-secao{font-size:.74rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
  color:var(--apagado);margin:0 0 .25em}
.stApp .e-tit{font-size:1.14rem;font-weight:700;font-stretch:82%;line-height:1.2;margin:0;color:var(--tinta)}
.stApp .e-painel{margin-bottom:.5rem}
.stApp .e-sub{font-size:.88rem;color:var(--apagado);margin:.3em 0 .1em;line-height:1.4}
.stApp .e-sub strong{color:var(--tinta2)}
/* placar dos cinco testes (Painel 3) */
.stApp table.placar{width:100%;border-collapse:collapse;border:none;margin:2px 0 4px;font-size:.92rem}
.stApp table.placar th{font-size:.78rem;font-weight:600;color:var(--apagado);text-align:left;padding:6px 10px;
  border:none;border-bottom:1px solid var(--fio);background:none}
.stApp table.placar td{padding:7px 10px;border:none;border-bottom:1px solid var(--fio);vertical-align:middle;
  color:var(--tinta)}
.stApp table.placar td.teste{font-weight:800;font-stretch:80%;font-size:1.05rem}
.stApp .selo{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;font-weight:650;white-space:nowrap}
.stApp .selo.rejeita{background:var(--ardosia);color:#fff}
.stApp .selo.nao-rejeita{background:#e3e7ec;color:var(--tinta2)}
.stApp table.placar td.acertou{color:var(--tinta);font-weight:700}
.stApp table.placar td.errou{color:var(--crit);font-weight:600}
.stApp table.placar td.inofensivo{color:var(--apagado);font-weight:600}
.stApp .placar-titulo{font-size:1.14rem;font-weight:700;font-stretch:82%;margin:.4em 0 0;color:var(--tinta)}
</style>"""


def cabecalho(sobretitulo: str, titulo: str, contexto: str | None = None) -> None:
    """Cabeçalho comum, no estilo da IA001: o logo da UFRGS uma vez, no canto da barra de
    navegação (`st.logo`); o estilo da tela (`CSS`); e, sem caixa, uma linha pequena com o
    nome da página (`sobretitulo`), o título forte com a pergunta da página, uma linha cinza
    de contexto e a autoria, discreta."""
    st.logo(_logo("ufrgs.svg"), size="large", link="https://www.ufrgs.br/")
    st.html(CSS)
    partes = [
        f'<p class="e-sobre">{html_inline(sobretitulo)}</p>',
        f'<p class="e-titulo" role="heading" aria-level="1">{html_inline(titulo)}</p>',
    ]
    if contexto:
        partes.append(f'<p class="e-contexto">{html_inline(contexto)}</p>')
    partes.append(f'<p class="e-autoria">{html_inline(AUTORIA)}</p>')
    _html(f'<header class="e-topo">{"".join(partes)}</header>')


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
    """Os números dos cartões: as duas proporções do Norte e Nordeste (contratos e valor), as
    diferenças de média de spread e de prazo (Norte e Nordeste − demais) e as médias de cada grupo."""
    nne = df["grupo"] == "N+NE"
    media = df.groupby("grupo")[["spread", "prazo_amortizacao"]].mean()
    return {
        "contratos": float(nne.mean()),
        "valor": float(df.loc[nne, "valor_contratado"].sum() / df["valor_contratado"].sum()),
        "spread": float(media.loc["N+NE", "spread"] - media.loc["Demais", "spread"]),
        "prazo": float(media.loc["N+NE", "prazo_amortizacao"] - media.loc["Demais", "prazo_amortizacao"]),
        "spread_nne": float(media.loc["N+NE", "spread"]),
        "spread_demais": float(media.loc["Demais", "spread"]),
        "prazo_nne": float(media.loc["N+NE", "prazo_amortizacao"]),
        "prazo_demais": float(media.loc["Demais", "prazo_amortizacao"]),
    }


def itens_descritivos(pop: pd.DataFrame, amostra: pd.DataFrame, qual: str) -> list[dict]:
    """Os quatro cartões dos Painéis 1 e 2 (ver `cartoes`). As duas proporções levam o medidor
    contra as réguas de H1 (PIB) e de H2 (população) e uma nota por régua; spread e prazo levam as
    duas médias e uma nota contra a promessa da política regional. No Painel 2 (`qual="amostra"`)
    o valor é o da amostra, e a linha logo abaixo dele diz a diferença para a população."""
    ind_pop, ind_amo = _indicadores(pop), _indicadores(amostra)
    ind = ind_amo if qual == "amostra" else ind_pop
    p0_pib, p0_pop = (h["p0_ou_delta0"] for h in c.HIPOTESES[:2])
    reguas = [(p0_pib, "PIB"), (p0_pop, "população")]
    # O mesmo fim de barra nos dois painéis e nos dois cartões de proporção.
    teto = 1.25 * max(p0_pib, p0_pop, *(i[k] for i in (ind_pop, ind_amo) for k in ("contratos", "valor")))

    def delta(chave: str, casas: int, unidade: str, escala: float = 1.0) -> list[str]:
        if qual != "amostra":
            return []
        return [f"{com_sinal(escala * (ind_amo[chave] - ind_pop[chave]), casas)} {unidade} da população"]

    def proporcao(chave: str) -> dict:
        return {
            "medidor": {"valor": ind[chave], "marcas": reguas, "teto": teto},
            "notas": [nota_regua(ind[chave], p0_pop, "da população"), nota_regua(ind[chave], p0_pib, "do PIB")],
        }

    return [
        {"rotulo": "Contratos em Norte e Nordeste", "valor": pct(ind["contratos"], 1),
         "sub": delta("contratos", 1, "p.p.", 100), **proporcao("contratos"),
         "ajuda": "Proporção dos contratos cuja UF principal fica no Norte ou no Nordeste. É o π de H1 e H2."},
        {"rotulo": "Valor contratado em Norte e Nordeste", "valor": pct(ind["valor"], 1),
         "sub": delta("valor", 1, "p.p.", 100), **proporcao("valor"),
         "ajuda": "A mesma proporção, mas somando reais em vez de contar contratos."},
        {"rotulo": "Spread: Norte e Nordeste − demais", "valor": f"{com_sinal(ind['spread'], 2)} p.p.",
         "sub": delta("spread", 2, "p.p.")
         + [f"médias: {br(ind['spread_nne'], 2)} contra {br(ind['spread_demais'], 2)} % a.a."],
         "notas": [nota_condicao(ind["spread"], ind["spread_demais"], -1, "um spread menor",
                                 QUASE_IGUAL * abs(ind["spread_demais"]))],
         "ajuda": "Diferença entre os spreads médios (% ao ano). Negativo: Norte e Nordeste pagam menos. É o δ de H3 e H4."},
        {"rotulo": "Prazo: Norte e Nordeste − demais", "valor": f"{com_sinal(ind['prazo'], 1)} meses",
         "sub": delta("prazo", 1, "meses")
         + [f"médias: {br(ind['prazo_nne'], 1)} contra {br(ind['prazo_demais'], 1)} meses"],
         "notas": [nota_condicao(ind["prazo"], ind["prazo_demais"], +1, "um prazo maior",
                                 QUASE_IGUAL * abs(ind["prazo_demais"]))],
         "ajuda": "Diferença entre os prazos médios de amortização. Positivo: Norte e Nordeste têm mais prazo. É o parâmetro de H5."},
    ]


def cartoes_descritivos(pop: pd.DataFrame, amostra: pd.DataFrame, qual: str) -> None:
    """Os quatro cartões dos Painéis 1 e 2 (`itens_descritivos`)."""
    cartoes(itens_descritivos(pop, amostra, qual))


# --- O painel descritivo -------------------------------------------------------------

def _unidade_maior(resumo: pd.DataFrame, campo: str, grupo: str | None = None) -> pd.Series:
    base = resumo if grupo is None else resumo[resumo["grupo"] == grupo]
    return base.sort_values(campo, ascending=False, kind="stable").iloc[0]


def _pontos(dif: float) -> str:
    """Distância em pontos percentuais, com a concordância: "0,5 ponto", "16,3 pontos"."""
    v = round(100 * abs(dif), 1)
    return f"{br(v, 1)} {'ponto' if v < 2 else 'pontos'}"


def titulo_onde(resumo: pd.DataFrame, unid: pd.DataFrame, nivel: str) -> str:
    """O título do bloco "Onde estão os contratos": a conclusão do mapa e do ranking, calculada
    do `resumo` (`graficos.por_unidade`) no nível pedido."""
    maior = _unidade_maior(resumo, "participacao")
    nne = resumo.loc[resumo["grupo"] == "N+NE", "participacao"].sum()
    if nivel == "regiao":
        return (f"O **{maior['nome']}** concentra **{pct(maior['participacao'])} dos contratos**; "
                f"Norte e Nordeste juntos ficam com **{pct(nne)}**.")
    n_nne = int((unid["grupo"] == "N+NE").sum())
    comparacao = "mais" if maior["participacao"] > nne else "menos"
    return (f"Só **{maior['nome']}** tem **{pct(maior['participacao'])} dos contratos**, {comparacao} "
            f"que os {n_nne} estados do Norte e do Nordeste juntos (**{pct(nne)}**).")


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
        nne = resumo.loc[resumo["grupo"] == "N+NE", "participacao"].sum()
        with bloco(titulo_onde(resumo, unid, nivel),
                   "Quanto mais escuro no mapa, maior a parte dos contratos; o ranking ao lado dá o número."):
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

        destaque = _unidade_maior(resumo, "participacao", "N+NE")
        p0_pib, p0_pop = (h["p0_ou_delta0"] for h in c.HIPOTESES[:2])
        soma_pop_exata = float(unid.loc[unid["grupo"] == "N+NE", "parte_populacao"].sum())
        with bloco(
            f"{destaque['nome']}: {pct(destaque['participacao'])} dos contratos, {pct(destaque['parte_pib'])} do PIB, "
            f"{pct(destaque['parte_populacao'])} da população.",
            f"Norte e Nordeste juntos têm {pct(nne)} dos contratos: a {_pontos(nne - p0_pib)} "
            f"da régua do PIB ({pct(p0_pib)}) e a {_pontos(nne - p0_pop)} da régua da população ({pct(p0_pop)})."
            + (f" A primeira linha do gráfico, {g.NOME_SOMA_NNE}, põe essa comparação lado a lado."
               if nivel == "regiao" else ""),
            secao="Três réguas: contratos, PIB e população",
        ):
            grafico(g.tres_reguas(df, unid, nivel, f"Contratos, PIB e população {por_onde}", max_x=max_reguas,
                                  legenda=(f"{base}. Ordem: % dos contratos, da maior para a menor.",),
                                  reguas_nne=(p0_pib, p0_pop)))
            st.caption(f"A régua do PIB ({pct(p0_pib)}) é a de H1; a da população ({pct(p0_pop)}), a de H2. "
                       "O pré-registro somou as partes já arredondadas, e a soma exata no IBGE dá "
                       f"{pct(soma_pop_exata, 2)}.")
            numeros(tabela_unidades(pop, comparar, unid, nivel,
                                    [("participacao", "% dos contratos", lambda v: pct(v, 1))], reguas=True))

        valor_nne = resumo.loc[resumo["grupo"] == "N+NE", "participacao_valor"].sum()
        maior_ou_menor = "maiores" if valor_nne > nne else "menores"
        with bloco(
            f"Somando reais em vez de contar contratos, Norte e Nordeste ficam com **{pct(valor_nne)} do valor**, "
            f"contra {pct(nne)} dos contratos: os contratos de lá são, em média, {maior_ou_menor}.",
            "Os testes contam contratos; o BNDES e o IBGE costumam falar em valor, e um contrato de R$ 1 mil pesa "
            "como um de R$ 10 bilhões na contagem.",
            secao="E em valor contratado?",
        ):
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
        medias = df.groupby("grupo")[["spread", "prazo_amortizacao"]].mean()
        s_nne, s_dem = medias.loc["N+NE", "spread"], medias.loc["Demais", "spread"]
        p_nne, p_dem = medias.loc["N+NE", "prazo_amortizacao"], medias.loc["Demais", "prazo_amortizacao"]
        if abs(s_nne - s_dem) < QUASE_IGUAL * s_dem:
            sobre_spread = f"O spread médio é **quase igual** nos dois grupos ({br(s_nne, 2)} contra {br(s_dem, 2)} % a.a.)"
        else:
            sobre_spread = (f"O spread médio é **{'menor' if s_nne < s_dem else 'maior'}** em Norte e Nordeste "
                            f"({br(s_nne, 2)} contra {br(s_dem, 2)} % a.a.)")
        sobre_prazo = (f"o prazo médio de amortização é **{'maior' if p_nne > p_dem else 'menor'}** em Norte e Nordeste "
                       f"({br(p_nne, 1)} contra {br(p_dem, 1)} meses)")
        with bloco(f"{sobre_spread}; {sobre_prazo}.",
                   f"Abaixo, a média de cada {onde}, com o número de contratos (n) ao lado: com poucos contratos, "
                   "a média oscila mais."):
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
            mediana = df.groupby("grupo")[coluna].median()
            with bloco(
                f"Metade dos contratos de Norte e Nordeste fica abaixo de **{br(mediana['N+NE'], casas)} {unidade}**, "
                f"contra **{br(mediana['Demais'], casas)} {unidade}** nas demais regiões (a mediana, o traço branco "
                "na caixa).",
                "O histograma mostra onde os contratos se concentram; a caixa vai do 1º ao 3º quartil.",
                secao=f"{titulo} por grupo: a distribuição inteira",
            ):
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

        infra = df.groupby("grupo")["setor"].apply(lambda s: (s == "Infraestrutura").mean())
        with bloco(
            f"Infraestrutura é **{pct(infra['N+NE'], 0)} dos contratos de Norte e Nordeste** e "
            f"{pct(infra['Demais'], 0)} nas demais regiões.",
            "Obras de infraestrutura têm prazos longos por natureza, então parte da diferença de prazo pode vir do "
            "setor, não da região (limitação 4 do Painel 3).",
            secao="Setor dos contratos em cada grupo",
        ):
            grafico(g.barras_setor(df, "Setor dentro de cada grupo", legenda=(f"{base}. Cada grupo soma 100%.",)))
            numeros(tabela_qualitativa_por_grupo(pop, comparar, "setor", "Setor"),
                    tabela_qualitativa_por_grupo(pop, comparar, "porte", "Porte"),
                    rotulo="Ver os números (setor e porte do cliente)")

    # ---------------------------------------------------------------- Qualidade
    with aba_qualidade:
        q = _contar_qualidade(df)
        sem = q["Sem spread em reais (fora de H3 e H4)"]
        with bloco(
            f"**{inteiro(sem)} contratos ({pct(sem / q['Contratos'])}) não têm subcrédito indexado em reais** e ficam "
            "fora do spread (H3, H4 e a descritiva do spread), mas continuam em H1, H2 e H5.",
            "Spreads e prazos iguais a 0, que parecem erro de registro, foram mantidos (limitação 5 do Painel 3).",
        ):
            cartoes([
                {"rotulo": "Contratos", "valor": inteiro(q["Contratos"]), "sub": base},
                {"rotulo": "Sem spread em reais", "valor": inteiro(sem), "sub": "ficam fora de H3 e H4",
                 "ajuda": "Ficam fora de H3 e H4."},
                {"rotulo": "Spread = 0", "valor": inteiro(q["Spread = 0"]), "sub": "mantidos (limitação 5)",
                 "ajuda": "Mantidos (limitação 5)."},
                {"rotulo": "Prazo de amortização = 0", "valor": inteiro(q["Prazo de amortização = 0"]),
                 "sub": "mantidos (limitação 5)", "ajuda": "Mantidos (limitação 5)."},
            ])
            numeros(tabela_qualidade(pop, comparar), rotulo="Ver os números por grupo")
