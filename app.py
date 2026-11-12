"""O BNDES compensa a desigualdade regional? App de quatro páginas.

Trabalho final de Probabilidade e Estatística: amostragem, intervalos de confiança e
testes de hipóteses sobre as operações não automáticas do BNDES (dados abertos).
Cada página tem URL própria, para abrir cada painel direto pelo link.

Executar, desta pasta ou de qualquer outra:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

PASTA = Path(__file__).resolve().parent / "paginas"

st.set_page_config(layout="wide", page_title="BNDES e desigualdade regional")

# A página padrão abre na raiz do app (o Streamlit ignora o `url_path` da página
# padrão); as outras três abrem em /populacao, /amostra e /inferencia.
paginas = [
    st.Page(PASTA / "base.py", title="Base e amostra", url_path="base", default=True),
    st.Page(PASTA / "populacao.py", title="Painel 1 — População", url_path="populacao"),
    st.Page(PASTA / "amostra.py", title="Painel 2 — Amostra", url_path="amostra"),
    st.Page(PASTA / "inferencia.py", title="Painel 3 — Inferência", url_path="inferencia"),
]
st.navigation(paginas).run()
