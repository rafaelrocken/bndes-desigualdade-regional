"""Painel 1: análise descritiva da população (os 6.320 contratos)."""
import sys
from pathlib import Path

# A página também roda sozinha (testes, `streamlit run paginas/populacao.py`): põe a raiz do app no caminho.
_RAIZ = str(Path(__file__).resolve().parents[1])
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

import streamlit as st  # noqa: E402

from dados_app import carregar  # noqa: E402
from tela import br, inteiro, painel_descritivo, pct  # noqa: E402

pop, amostra, meta = carregar()

st.title("Painel 1 — População")
nne = pop["grupo"] == "N+NE"
valor_nne = pop.loc[nne, "valor_contratado"].sum() / pop["valor_contratado"].sum()
infra = pop.groupby("grupo")["setor"].apply(lambda s: (s == "INFRAESTRUTURA").mean())
prazo = pop.groupby("grupo")["prazo_amortizacao"].mean()
spread = pop.groupby("grupo")["spread"].mean()
st.markdown(
    f"Todos os **{inteiro(len(pop))} contratos** da população. Os números aqui são os **valores "
    "verdadeiros**, que o Painel 3 usa para conferir a inferência feita com a amostra."
)
st.markdown(
    f"""
**Em poucas linhas.**
- N+NE têm **{pct(nne.mean())} dos contratos**, mas **{pct(valor_nne)} do valor contratado**: os contratos de lá são, em média, maiores.
- O spread médio é quase igual nos dois grupos ({br(spread["N+NE"], 2)} contra {br(spread["Demais"], 2)} % a.a.).
- O prazo de amortização médio é bem maior em N+NE ({inteiro(prazo["N+NE"])} contra {inteiro(prazo["Demais"])} meses), onde a infraestrutura pesa mais ({pct(infra["N+NE"], 0)} dos contratos, contra {pct(infra["Demais"], 0)} nos Demais).
"""
)

painel_descritivo(pop, amostra, "populacao")
