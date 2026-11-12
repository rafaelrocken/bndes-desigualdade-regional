"""Painel 2: análise descritiva da amostra (os 913 contratos sorteados), ao lado da população."""
import sys
from pathlib import Path

# A página também roda sozinha (testes, `streamlit run paginas/amostra.py`): põe a raiz do app no caminho.
_RAIZ = str(Path(__file__).resolve().parents[1])
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

import streamlit as st  # noqa: E402

import calculo as c  # noqa: E402
from dados_app import carregar  # noqa: E402
from tela import GRUPOS, br, inteiro, painel_descritivo  # noqa: E402

pop, amostra, meta = carregar()

st.title("Painel 2 — Amostra")
st.markdown(
    f"Os **{inteiro(len(amostra))} contratos** sorteados (amostra aleatória simples, semente "
    f"{meta['semente']}), com os mesmos gráficos do Painel 1, nas mesmas escalas. Cada tabela traz a "
    "coluna **População** ao lado da **Amostra**, para mostrar onde as duas se afastam."
)

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
st.markdown(
    f"**A amostra se parece com a população?** Sim: nas proporções por região ela fica a no máximo "
    f"{br(100 * dif_reg.max(), 1)} ponto percentual da população ({regiao_max}), e médias, medianas e "
    "quartis de spread e prazo andam juntos nos dois conjuntos. A maior diferença está no "
    f"{nome} do grupo {grupo} ({br(m_amo, casas)} contra {br(m_pop, casas)} {unidade} na população), "
    "efeito esperado da variabilidade amostral, que pesa mais no grupo menor."
)

painel_descritivo(pop, amostra, "amostra")
