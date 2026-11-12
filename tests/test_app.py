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
    frase = next(m.value for m in at.markdown if "**A amostra se parece com a população?**" in m.value)
    assert "valor" in frase and "R$" in frase.replace("\\$", "$")
    assert "Sudeste" in frase  # a região com a maior diferença na participação por valor
    assert "N+NE" not in frase  # nomes por extenso no texto
    # o grupo citado na maior diferença de spread/prazo é o mesmo cuja média aparece entre parênteses
    import re

    import pandas as pd

    from tela import br

    amostra = pd.read_parquet(RAIZ / "dados" / "amostra.parquet")
    achado = re.search(r"no (spread|prazo médio) do grupo (Norte e Nordeste|Demais regiões) \(([^ ]+) contra", frase)
    assert achado, frase
    coluna = "spread" if achado.group(1) == "spread" else "prazo_amortizacao"
    casas = 2 if coluna == "spread" else 1
    grupo = {"Norte e Nordeste": "N+NE", "Demais regiões": "Demais"}[achado.group(2)]
    media = amostra.loc[amostra["grupo"] == grupo, coluna].mean()
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


# ---------------------------------------------------------------- Task 7b: redesenho

import json  # noqa: E402
import re  # noqa: E402

import pyarrow as pa  # noqa: E402

import graficos as g  # noqa: E402


def _specs(at: AppTest) -> list[dict]:
    """A especificação Vega-Lite de cada gráfico da página, na ordem."""
    return [json.loads(e.proto.spec) for e in at.get("vega_lite_chart")]


def _titulo_do_grafico(spec: dict) -> str:
    titulo = spec.get("title", {})
    return titulo.get("text", "") if isinstance(titulo, dict) else str(titulo)


def _linhas_do_grafico(at: AppTest, comeco_do_titulo: str) -> int:
    """Número de linhas dos dados do primeiro gráfico cujo título começa com `comeco_do_titulo`."""
    for elemento in at.get("vega_lite_chart"):
        if _titulo_do_grafico(json.loads(elemento.proto.spec)).startswith(comeco_do_titulo):
            return pa.ipc.open_stream(elemento.proto.datasets[0].data.data).read_all().num_rows
    raise AssertionError(f"nenhum gráfico com título começando por {comeco_do_titulo!r}")


def _escalas(no) -> list:
    """Todas as escalas de eixo e de cor declaradas num spec, na ordem em que aparecem."""
    achadas = []
    if isinstance(no, dict):
        for chave, valor in no.items():
            if chave == "scale" and isinstance(valor, dict):
                achadas.append(json.dumps(valor, sort_keys=True))
            else:
                achadas += _escalas(valor)
    elif isinstance(no, list):
        for item in no:
            achadas += _escalas(item)
    return achadas


def _com_nivel(relativo: str, nivel: str) -> AppTest:
    at = AppTest.from_file(str(RAIZ / relativo), default_timeout=TEMPO)
    at.query_params["ver"] = nivel
    at.run()
    if relativo in ("paginas/populacao.py", "paginas/amostra.py"):
        assert at.session_state["_nivel"] == nivel  # o modo pedido na URL foi mesmo aplicado
    return at


@pytest.mark.parametrize("pagina", ["paginas/populacao.py", "paginas/amostra.py"])
def test_seletor_muda_o_ranking_de_5_para_27_barras(pagina):
    at = _rodar(pagina)
    seletor = at.button_group[0]
    assert seletor.value == "regiao"  # o padrão é Região
    assert _linhas_do_grafico(at, "Ranking") == 5
    seletor.set_value("estado").run()
    assert not at.exception
    assert _linhas_do_grafico(at, "Ranking") == 27


def test_seletor_aceita_parametro_de_url():
    at = _com_nivel("paginas/populacao.py", "estado")
    assert at.button_group[0].value == "estado"
    assert _linhas_do_grafico(at, "Ranking") == 27


def test_seletor_vale_nos_dois_paineis():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=TEMPO).run()
    at.switch_page("paginas/populacao.py").run()
    at.button_group[0].set_value("estado").run()
    at.switch_page("paginas/amostra.py").run()
    assert not at.exception
    assert at.session_state["_nivel"] == "estado"
    assert at.button_group[0].value == "estado"
    assert _linhas_do_grafico(at, "Ranking") == 27


@pytest.mark.parametrize("nivel", ["regiao", "estado"])
def test_mapa_tem_o_mesmo_dominio_de_cor_nos_dois_paineis(nivel):
    mapas = []
    for pagina in ("paginas/populacao.py", "paginas/amostra.py"):
        spec = next(s for s in _specs(_com_nivel(pagina, nivel)) if _titulo_do_grafico(s).startswith("% dos contratos"))
        mapas.append(spec["encoding"]["color"]["scale"])
    assert mapas[0] == mapas[1]
    assert mapas[0]["domain"] == g.FAIXAS_MAPA  # sequencial, em faixas de limiares fixos


@pytest.mark.parametrize("nivel", ["regiao", "estado"])
def test_paineis_1_e_2_tem_as_mesmas_escalas_em_todos_os_graficos(nivel):
    """Mesmos gráficos, na mesma ordem, com as mesmas escalas (eixos e cores) nos dois painéis."""
    specs_pop = _specs(_com_nivel("paginas/populacao.py", nivel))
    specs_amo = _specs(_com_nivel("paginas/amostra.py", nivel))
    assert len(specs_pop) == len(specs_amo) >= 12
    for pop, amo in zip(specs_pop, specs_amo):
        assert _escalas(pop) == _escalas(amo), _titulo_do_grafico(pop)


@pytest.mark.parametrize("pagina", PAGINAS)
@pytest.mark.parametrize("nivel", ["regiao", "estado"])
def test_nenhuma_pagina_tem_excecao_nos_dois_modos(pagina, nivel):
    at = _com_nivel(pagina, nivel)
    assert not at.exception
    assert not at.warning


@pytest.mark.parametrize("pagina", ["paginas/populacao.py", "paginas/amostra.py"])
def test_paineis_descritivos_tem_as_tres_abas_e_os_numeros_recolhidos(pagina):
    at = _rodar(pagina)
    assert [t.label for t in at.tabs] == ["Onde estão os contratos", "Condições: spread e prazo", "Qualidade dos dados"]
    assert at.expander and all(not e.proto.expanded for e in at.expander)
    assert all(e.label.startswith("Ver os números") for e in at.expander)


def test_painel_3_tem_uma_aba_por_hipotese():
    rotulos = [t.label for t in _rodar("paginas/inferencia.py").tabs]
    for hid in ["H1", "H2", "H3", "H4", "H5"]:
        assert sum(r.startswith(hid) for r in rotulos) == 1, hid
    for aba in ["Comparação com a população", "Variabilidade (1.000 sorteios)", "Limitações"]:
        assert aba in rotulos


def test_painel_3_respostas_batem_com_a_decisao_de_cada_teste():
    """A resposta destacada de cada aba diz "rejeita H0" ou "não rejeita H0" conforme o teste,
    sem casar "não rejeita" por dentro de "rejeita"."""
    import pandas as pd

    import calculo as c

    amostra = pd.read_parquet(RAIZ / "dados" / "amostra.parquet")
    n, x = len(amostra), int((amostra["grupo"] == "N+NE").sum())
    respostas = [i.value for i in _rodar("paginas/inferencia.py").info if "o teste" in i.value]
    assert len(respostas) == 5
    for h, texto in zip(c.HIPOTESES, respostas):
        if h["variavel"] == "grupo":
            rejeita = c.teste_z_proporcao(x, n, h["p0_ou_delta0"])["rejeita"]
        else:
            a = amostra.loc[amostra["grupo"] == "N+NE", h["variavel"]]
            b = amostra.loc[amostra["grupo"] == "Demais", h["variavel"]]
            rejeita = c.teste_welch(a, b, h["alternativa"])["rejeita"]
        nao_rejeita = re.search(r"\bnão rejeita H0\b", texto) is not None
        rejeita_sozinho = re.search(r"(?<!não )\brejeita H0\b", texto) is not None
        assert (rejeita_sozinho, nao_rejeita) == (rejeita, not rejeita), (h["id"], texto)


def test_painel_3_ajustes_da_revisao():
    at = _rodar("paginas/inferencia.py")
    texto = _textos(at)
    assert "coerente com não rejeitar H0" in texto
    assert "coerente com não rejeita H0" not in texto
    assert "≈ 100%" in texto and "100,0%" not in texto
    assert "de um desvio padrão" in texto  # o gloss do d de Cohen em H3
    itens = " ".join(" ".join(map(str, t.value.to_numpy().ravel())) for t in at.table)
    assert "Estimativa − verdadeiro" in itens
    # a justificativa do parâmetro de média (auditor), antes das abas
    assert any("diferença de médias populacionais" in i.value for i in at.info)


def test_painel_3_grafico_da_simulacao_tem_dados():
    at = _rodar("paginas/inferencia.py")
    elemento = next(e for e in at.get("vega_lite_chart")
                    if _titulo_do_grafico(json.loads(e.proto.spec)).startswith("p̂ em"))
    tabelas = [pa.ipc.open_stream(d.data.data).read_all().to_pandas() for d in elemento.proto.datasets]
    barras = next(t for t in tabelas if "sorteios" in t.columns)
    linhas = next(t for t in tabelas if "referencia" in t.columns)
    assert barras["sorteios"].sum() == 1000
    assert len(linhas) == 3


def test_setores_com_acento_nas_tabelas():
    at = _rodar("paginas/populacao.py")
    celulas = " ".join(" ".join(map(str, d.value.to_numpy().ravel())) for d in at.dataframe)
    for setor in ["Indústria", "Comércio e serviços", "Agropecuária", "Infraestrutura"]:
        assert setor in celulas
    assert "INDUSTRIA" not in celulas and "COMERCIO/SERVICOS" not in celulas


@pytest.mark.parametrize("pagina", PAGINAS)
def test_cada_pagina_abre_com_a_pergunta_e_os_numeros_grandes(pagina):
    at = _rodar(pagina)
    assert at.markdown[0].value.startswith("#### ") and at.markdown[0].value.endswith("?")
    assert 3 <= len(at.metric) - (4 if "populacao" in pagina or "amostra" in pagina else 0) <= 4


def test_painel_2_cartoes_tem_a_diferenca_para_a_populacao():
    cartoes = _rodar("paginas/amostra.py").metric[:4]
    assert all("da população" in m.delta for m in cartoes)


def _links(at: AppTest) -> list[tuple[str, str]]:
    return [(e.proto.page, e.proto.label) for e in at.get("page_link")]


def test_raiz_tem_links_para_os_tres_paineis():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=TEMPO).run()
    assert at.title[0].value == "Base de dados e amostra"
    paginas = [pagina for pagina, _ in _links(at)]
    assert paginas == ["populacao", "amostra", "inferencia"]
    assert "quatro partes" in _textos(at)


@pytest.mark.parametrize("pagina", ["paginas/populacao.py", "paginas/amostra.py", "paginas/inferencia.py"])
def test_cada_painel_tem_link_de_volta_para_a_base(pagina):
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=TEMPO).run()
    at.switch_page(pagina).run()
    assert ("", "Base de dados e amostra") in _links(at)  # a página da base fica na raiz
