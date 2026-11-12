"""Testes do app de quatro páginas, com o AppTest do Streamlit (sem navegador).

Cada página roda sozinha (o AppTest executa o arquivo da página como um script) e
o app inteiro roda pelo `app.py`, inclusive a partir de outra pasta.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = ["paginas/base.py", "paginas/populacao.py", "paginas/amostra.py", "paginas/inferencia.py"]
TEMPO = 60


def _rodar(relativo: str) -> AppTest:
    return AppTest.from_file(str(RAIZ / relativo), default_timeout=TEMPO).run()


def _textos(at: AppTest) -> str:
    """Todo o texto visível de markdown, legendas, avisos e títulos de expansores."""
    partes = [m.value for m in at.markdown]
    partes += [c.value for c in at.caption]
    partes += [i.value for i in at.info] + [w.value for w in at.warning]
    partes += [e.label for e in at.expander]
    partes += [t.value for t in at.title] + [h.value for h in at.header] + [s.value for s in at.subheader]
    return " ".join(partes)


def _graficos(at: AppTest) -> int:
    """Quantos gráficos Altair/Vega-Lite a página desenhou."""
    return len(at.get("vega_lite_chart"))


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_abre_sozinha(pagina):
    at = _rodar(pagina)
    assert not at.exception
    assert not at.warning  # nenhum aviso de depreciação nem de erro na tela


def test_navegacao_a_partir_de_outro_diretorio(monkeypatch):
    monkeypatch.chdir(RAIZ.parent)
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=TEMPO).run()
    assert not at.exception
    # a página padrão é a da base
    assert "6.320" in _textos(at)


@pytest.mark.parametrize(
    "pagina, titulo",
    [("paginas/populacao.py", "Painel 1"), ("paginas/amostra.py", "Painel 2"), ("paginas/inferencia.py", "Painel 3")],
)
def test_navegacao_troca_de_pagina(pagina, titulo):
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=TEMPO).run()
    at.switch_page(pagina).run()
    assert not at.exception
    assert at.title[0].value.startswith(titulo)


def test_inferencia_mostra_cinco_hipoteses():
    at = _rodar("paginas/inferencia.py")
    texto = " ".join(m.value for m in at.markdown)
    assert all(h in texto for h in ["H1", "H2", "H3", "H4", "H5"]) and "aceita" not in texto.lower()


def test_inferencia_sem_frases_proibidas():
    texto = _textos(_rodar("paginas/inferencia.py")).lower()
    assert "aceita" not in texto
    assert "95% de chance" not in texto and "95 % de chance" not in texto
    # as decisões calculadas ao vivo aparecem em português
    assert "não rejeita h0" in texto and "rejeita h0" in texto


def test_inferencia_tem_as_oito_limitacoes():
    texto = _textos(_rodar("paginas/inferencia.py"))
    assert "O teste conta contratos; o BNDES e o IBGE falam em valor." in texto
    assert "Fração amostrada de 14%." in texto


def test_paineis_1_e_2_desenham_o_mesmo_numero_de_graficos():
    n_pop = _graficos(_rodar("paginas/populacao.py"))
    n_amo = _graficos(_rodar("paginas/amostra.py"))
    assert n_pop >= 6
    assert n_pop == n_amo


def test_base_mostra_tamanhos():
    texto = _textos(_rodar("paginas/base.py"))
    assert "913" in texto
    assert "6.320" in texto or "6320" in texto
    assert "20261112" in texto and "c47f903" in texto


def test_painel_1_mostra_qualidade_dos_dados():
    at = _rodar("paginas/populacao.py")
    # os rótulos da tabela de qualidade estão na primeira coluna ("Item")
    celulas = " ".join(" ".join(map(str, d.value.to_numpy().ravel())) for d in at.dataframe).lower()
    assert "sem spread em reais" in celulas
    assert "spread = 0" in celulas
    assert "prazo de amortização = 0" in celulas


def test_base_tabelas_de_texto_longo_nao_cortam():
    """Variáveis e regras de agregação vão em `st.table` (quebra linha), não em `st.dataframe`."""
    at = _rodar("paginas/base.py")
    estaticas = " ".join(" ".join(map(str, t.value.to_numpy().ravel())) for t in at.table)
    assert "TAXA FIXA EM US$" in estaticas  # a regra longa do spread
    assert "qualitativa nominal" in estaticas  # a tabela de variáveis
    interativas = " ".join(" ".join(map(str, d.value.to_numpy().ravel())) for d in at.dataframe)
    assert "TAXA FIXA EM US$" not in interativas


def test_painel_2_fala_do_valor():
    """A frase de semelhança cobre também o valor, onde estão as maiores diferenças."""
    at = _rodar("paginas/amostra.py")
    frase = next(m.value for m in at.markdown if "se parece com a população" in m.value)
    assert "valor" in frase and "R$" in frase.replace("\\$", "$")
    assert "SE" in frase  # a região com a maior diferença na participação por valor
    # o grupo citado na maior diferença de spread/prazo é o mesmo cuja média aparece entre parênteses
    import re

    import pandas as pd

    from tela import br

    amostra = pd.read_parquet(RAIZ / "dados" / "amostra.parquet")
    achado = re.search(r"no (spread|prazo médio) do grupo (N\+NE|Demais) \(([^ ]+) contra", frase)
    assert achado, frase
    coluna = "spread" if achado.group(1) == "spread" else "prazo_amortizacao"
    casas = 2 if coluna == "spread" else 1
    media = amostra.loc[amostra["grupo"] == achado.group(2), coluna].mean()
    assert achado.group(3) == br(media, casas)


def test_inferencia_mostra_a_simulacao():
    at = _rodar("paginas/inferencia.py")
    texto = _textos(at)
    assert "1.000 sorteios" in texto
    celulas = " ".join(" ".join(map(str, d.value.to_numpy().ravel())) for d in [*at.dataframe, *at.table])
    assert "Poder pela fórmula" in " ".join(" ".join(map(str, d.value.columns)) for d in [*at.dataframe, *at.table]) \
        or "Poder pela fórmula" in celulas
    # 6 QQ-plots (H3, H4, H5) + intervalos + o histograma de p̂ nos 1.000 sorteios
    assert _graficos(at) == 8


def test_painel_2_tem_coluna_populacao():
    at = _rodar("paginas/amostra.py")
    colunas = [" ".join(map(str, d.value.columns)) for d in at.dataframe]
    assert colunas and all("População" in c for c in colunas)
