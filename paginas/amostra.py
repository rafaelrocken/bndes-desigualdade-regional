"""Painel 2: análise descritiva da amostra (os 913 contratos sorteados), ao lado da população."""
import sys
from pathlib import Path

# A página também roda sozinha (testes, `streamlit run paginas/amostra.py`): põe a raiz do app no caminho.
_RAIZ = str(Path(__file__).resolve().parents[1])
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

import streamlit as st  # noqa: E402

import calculo as c  # noqa: E402
from dados_app import carregar, carregar_ibge, carregar_malhas  # noqa: E402
from tela import (  # noqa: E402
    GRUPOS,
    NOME_GRUPO,
    bloco,
    br,
    cartoes_descritivos,
    inteiro,
    painel_descritivo,
    seletor_nivel,
    voltar_para_base,
    cabecalho,
    rodape,
)
from graficos import NOME_REGIAO  # noqa: E402

pop, amostra, meta = carregar()

voltar_para_base()
cabecalho(
    "Painel 2 · Amostra",
    f"A amostra de {inteiro(len(amostra))} contratos sorteados se parece com a população?",
    f"Os **{inteiro(len(amostra))} contratos** sorteados (amostra aleatória simples, semente "
    f"{meta['semente']}), com os mesmos gráficos do Painel 1, nas mesmas escalas. Cada cartão mostra o valor "
    "da amostra e, abaixo, a diferença para a população; cada tabela traz a coluna **População** ao lado da "
    "**Amostra**.",
)
cartoes_descritivos(pop, amostra, "amostra")

# A amostra se parece com a população? Diferenças medidas aqui, ao vivo.
p_reg_amo = amostra["regiao"].value_counts(normalize=True)
p_reg_pop = pop["regiao"].value_counts(normalize=True)
dif_reg = (p_reg_amo.reindex(p_reg_pop.index, fill_value=0) - p_reg_pop).abs()
regiao_max = dif_reg.idxmax()

maior = None  # (diferença relativa, variável, grupo, média da amostra, média da população)
for coluna, nome, unidade in [("spread", "spread", "% a.a."), ("prazo_amortizacao", "prazo médio", "meses")]:
    for grupo in GRUPOS:
        m_amo = c.descritiva(amostra.loc[amostra["grupo"] == grupo, coluna])["media"]
        m_pop = c.descritiva(pop.loc[pop["grupo"] == grupo, coluna])["media"]
        relativa = abs(m_amo - m_pop) / abs(m_pop)
        if maior is None or relativa > maior[0]:
            maior = (relativa, nome, unidade, grupo, m_amo, m_pop)
_, nome, unidade, grupo, m_amo, m_pop = maior
casas = 2 if unidade == "% a.a." else 1


# O valor contratado: participação por região em valor e valor médio de cada grupo.
def _part_valor(df):
    """Participação de cada região no valor contratado (soma 1)."""
    soma = df.groupby("regiao")["valor_contratado"].sum()
    return soma / soma.sum()


dif_val = (_part_valor(amostra).reindex(p_reg_pop.index, fill_value=0) - _part_valor(pop)).abs()
regiao_val = dif_val.idxmax()
val_amo, val_pop = _part_valor(amostra).get(regiao_val, 0.0), _part_valor(pop)[regiao_val]
maior_val = None  # (diferença relativa, grupo, média da amostra, média da população)
for grupo_valor in GRUPOS:
    m_amo_v = amostra.loc[amostra["grupo"] == grupo_valor, "valor_contratado"].mean()
    m_pop_v = pop.loc[pop["grupo"] == grupo_valor, "valor_contratado"].mean()
    relativa = abs(m_amo_v - m_pop_v) / m_pop_v
    if maior_val is None or relativa > maior_val[0]:
        maior_val = (relativa, grupo_valor, m_amo_v, m_pop_v)
rel_v, grupo_v, m_amo_v, m_pop_v = maior_val
max_pop, max_amo = pop["valor_contratado"].max(), amostra["valor_contratado"].max()
maior_sorteado = bool((amostra["valor_contratado"] == max_pop).any())

COMPARACAO = (
    f"**A amostra se parece com a população?** Na contagem de contratos e nas condições, sim: nas "
    f"proporções por região ela fica a no máximo {br(100 * dif_reg.max(), 1)} ponto percentual da "
    f"população ({NOME_REGIAO[regiao_max]}), e médias, medianas e quartis de spread e prazo andam juntos; entre essas "
    f"medidas, a maior diferença está no {nome} do grupo {NOME_GRUPO[grupo]} ({br(m_amo, casas)} contra "
    f"{br(m_pop, casas)} {unidade} na população), efeito esperado da variabilidade amostral, que pesa "
    "mais no grupo menor. No **valor contratado**, bem menos: a participação por valor chega a se afastar "
    f"{br(100 * dif_val.max(), 1)} pontos percentuais da população ({NOME_REGIAO[regiao_val]}: {br(100 * val_amo, 1)}% "
    rf"contra {br(100 * val_pop, 1)}%), e o valor médio do grupo {NOME_GRUPO[grupo_v]} é de R\$ {br(m_amo_v / 1e6, 1)} "
    rf"milhões na amostra contra R\$ {br(m_pop_v / 1e6, 1)} milhões na população "
    f"({br(100 * rel_v, 0)}% de diferença). O valor tem cauda pesada: o maior contrato da população "
    rf"(R\$ {br(max_pop / 1e9, 1)} bilhões) {'entrou' if maior_sorteado else 'não entrou'} na amostra, cujo "
    rf"maior contrato é de R\$ {br(max_amo / 1e9, 1)} bilhões. Por isso a leitura por valor (limitação 1 "
    "do Painel 3) é muito mais ruidosa numa amostra de contratos do que a leitura por contagem."
)
# A resposta curta fica à vista, como título; o parágrafo inteiro, com os números, desce para um expansor.
with bloco(
    "Na contagem de contratos e nas condições, a amostra se parece com a população; no valor contratado, bem menos.",
    f"Por região, a amostra fica a no máximo {br(100 * dif_reg.max(), 1)} ponto percentual da população na "
    f"contagem, e a até {br(100 * dif_val.max(), 1)} pontos no valor.",
    secao="A amostra se parece com a população?",
):
    with st.expander("Ver a comparação por extenso"):
        st.markdown(COMPARACAO)

nivel = seletor_nivel("amostra")
painel_descritivo(pop, amostra, "amostra", carregar_ibge(), carregar_malhas(), nivel)

rodape()
