"""Testes dos gráficos do app.

Os gráficos são conferidos pela especificação Vega-Lite (`to_dict()`), sem
renderizar: escala, ordem das barras, cores por grupo, fonte no título e dados
que o gráfico carrega.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import _encodings

from graficos import (
    CORES,
    FONTE,
    barras_regiao,
    barras_setor,
    boxplot,
    dominio,
    histograma,
    intervalos,
    qqplot,
)

PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"


def _tabelas(spec: dict) -> list[list[dict]]:
    """Os conjuntos de dados embutidos na especificação, na ordem de criação."""
    return list(spec.get("datasets", {}).values())


def _cores_do_grupo(spec: dict) -> list[dict]:
    """Escalas de cor que usam o campo `grupo`."""
    return [c["scale"] for c in _encodings(spec) if c.get("field") == "grupo" and "scale" in c]


# ---------------------------------------------------------------- os dois do plano


def test_barras_comecam_em_zero_e_tem_fonte(pop_pequena):
    d = barras_regiao(pop_pequena, "contratos", "t").to_dict()
    assert d["encoding"]["x"]["scale"]["domain"][0] == 0 or d["encoding"]["y"]["scale"]["domain"][0] == 0
    # ensure_ascii=False: o padrão do json escapa "ç" e "ã" e a FONTE nunca seria achada
    assert FONTE in json.dumps(d, ensure_ascii=False)


def test_histograma_usa_dominio_dado(pop_pequena):
    d = histograma(pop_pequena, "spread", (0, 6), "t").to_dict()
    assert [0, 6] in [v.get("scale", {}).get("domain") for v in _encodings(d)]


# ---------------------------------------------------------------- constantes


def test_cores_okabe_ito_e_fonte():
    assert CORES == {"N+NE": "#D55E00", "Demais": "#0072B2"}
    assert FONTE == "Fonte: BNDES, operações não automáticas (dados abertos, baixado em 12/11/2026)"


# ---------------------------------------------------------------- barras_regiao


def test_barras_regiao_contratos_ordem_decrescente_e_eixo_de_0_a_100(pop_pequena):
    d = barras_regiao(pop_pequena, "contratos", "t").to_dict()
    # SE 7, NE 5, S 4, CO 3, N 1
    assert d["encoding"]["y"]["sort"] == ["SE", "NE", "S", "CO", "N"]
    assert d["encoding"]["x"]["scale"]["domain"] == [0, 1]
    assert d["encoding"]["x"]["axis"]["format"] == ".0%"
    participacao = {r["regiao"]: r["participacao"] for r in _tabelas(d)[0]}
    assert participacao["SE"] == pytest.approx(7 / 20)
    assert sum(participacao.values()) == pytest.approx(1.0)


def test_barras_regiao_valor_soma_um_e_mesma_escala(pop_pequena):
    d = barras_regiao(pop_pequena, "valor", "t").to_dict()
    linhas = _tabelas(d)[0]
    assert sum(r["participacao"] for r in linhas) == pytest.approx(1.0)
    esperado = pop_pequena.groupby("regiao")["valor_contratado"].sum()
    esperado = esperado / esperado.sum()
    for linha in linhas:
        assert linha["participacao"] == pytest.approx(esperado[linha["regiao"]])
    # a ordem acompanha a medida mostrada, do maior para o menor
    ordem = d["encoding"]["y"]["sort"]
    assert ordem == list(esperado.sort_values(ascending=False).index)
    # a escala é a mesma da medida "contratos": de 0 a 100%
    assert d["encoding"]["x"]["scale"]["domain"] == [0, 1]


def test_barras_regiao_mostra_cinco_regioes_mesmo_sem_contratos(pop_pequena):
    sem_norte = pop_pequena[pop_pequena["regiao"] != "N"]
    d = barras_regiao(sem_norte, "contratos", "t").to_dict()
    por_regiao = {r["regiao"]: r["participacao"] for r in _tabelas(d)[0]}
    assert set(por_regiao) == {"N", "NE", "CO", "SE", "S"}
    assert por_regiao["N"] == 0
    assert d["encoding"]["y"]["sort"][-1] == "N"


def test_barras_regiao_cor_pelo_grupo_da_regiao(pop_pequena):
    d = barras_regiao(pop_pequena, "contratos", "t").to_dict()
    grupo_da = {r["regiao"]: r["grupo"] for r in _tabelas(d)[0]}
    assert grupo_da == {"N": "N+NE", "NE": "N+NE", "CO": "Demais", "SE": "Demais", "S": "Demais"}
    escala = d["encoding"]["color"]["scale"]
    assert escala["domain"] == ["N+NE", "Demais"]
    assert escala["range"] == [CORES["N+NE"], CORES["Demais"]]
    assert d["encoding"]["color"]["legend"]["title"]


def test_barras_regiao_medida_invalida(pop_pequena):
    with pytest.raises(ValueError, match="medida"):
        barras_regiao(pop_pequena, "quantidade", "t")


# ---------------------------------------------------------------- barras_setor


def test_barras_setor_participacao_soma_um_por_grupo(pop_pequena):
    d = barras_setor(pop_pequena, "t").to_dict()
    tabela = pd.DataFrame(_tabelas(d)[0])
    assert tabela.groupby("grupo")["participacao"].sum().to_dict() == pytest.approx(
        {"N+NE": 1.0, "Demais": 1.0}
    )
    assert d["encoding"]["x"]["scale"]["domain"] == [0, 1]
    assert d["encoding"]["x"]["axis"]["format"] == ".0%"


def test_barras_setor_ordem_pela_participacao_geral_e_cores_fixas(pop_pequena):
    d = barras_setor(pop_pequena, "t").to_dict()
    esperado = list(pop_pequena["setor"].value_counts().sort_values(ascending=False, kind="stable").index)
    assert d["encoding"]["y"]["sort"] == esperado
    assert d["encoding"]["color"]["scale"] == {
        "domain": ["N+NE", "Demais"],
        "range": [CORES["N+NE"], CORES["Demais"]],
    }
    assert d["encoding"]["yOffset"]["field"] == "grupo"
    assert d["encoding"]["yOffset"]["sort"] == ["N+NE", "Demais"]


def test_barras_setor_grupo_ausente_nao_quebra(pop_pequena):
    so_demais = pop_pequena[pop_pequena["grupo"] == "Demais"]
    d = barras_setor(so_demais, "t").to_dict()
    tabela = pd.DataFrame(_tabelas(d)[0])
    assert tabela.groupby("grupo")["participacao"].sum()["Demais"] == pytest.approx(1.0)


# ---------------------------------------------------------------- histograma


def test_histograma_descarta_nan_e_conta_o_resto(pop_pequena):
    d = histograma(pop_pequena, "spread", (0, 6), "t").to_dict()
    tabela = pd.DataFrame(_tabelas(d)[0])
    assert tabela["contratos"].sum() == pop_pequena["spread"].notna().sum()  # 17 de 20
    # proporção dentro de cada grupo: soma 1 porque todos os valores cabem em [0, 6]
    assert tabela.groupby("grupo")["proporcao"].sum().to_dict() == pytest.approx(
        {"N+NE": 1.0, "Demais": 1.0}
    )
    assert len(tabela) == 2 * 30


def test_histograma_escala_x_com_clamp_e_bins(pop_pequena):
    d = histograma(pop_pequena, "spread", (0, 6), "t", bins=10).to_dict()
    x = d["spec"]["encoding"]["x"]
    assert x["scale"]["domain"] == [0, 6]
    assert x["scale"]["clamp"] is True
    assert len(_tabelas(d)[0]) == 2 * 10


def test_histograma_facetas_por_grupo_com_cores_fixas(pop_pequena):
    d = histograma(pop_pequena, "prazo_amortizacao", (0, 120), "t").to_dict()
    assert d["facet"]["row"]["field"] == "grupo"
    assert d["facet"]["row"]["sort"] == ["N+NE", "Demais"]
    assert d["spec"]["encoding"]["color"]["scale"] == {
        "domain": ["N+NE", "Demais"],
        "range": [CORES["N+NE"], CORES["Demais"]],
    }
    # a escala de y é compartilhada entre os grupos, para comparar as alturas
    assert d.get("resolve", {}).get("scale", {}).get("y", "shared") == "shared"


def test_histograma_eixos_em_portugues_com_unidade(pop_pequena):
    spread = histograma(pop_pequena, "spread", (0, 6), "t").to_dict()
    prazo = histograma(pop_pequena, "prazo_amortizacao", (0, 120), "t").to_dict()
    assert "% a.a." in spread["spec"]["encoding"]["x"]["title"]
    assert "meses" in prazo["spec"]["encoding"]["x"]["title"]
    titulo_y = spread["spec"]["encoding"]["y"]["title"]  # em duas linhas, uma lista
    assert "proporção" in " ".join(titulo_y).lower()


def test_histograma_rotulo_proprio(pop_pequena):
    d = histograma(pop_pequena, "spread", (0, 6), "t", rotulo="Meu eixo").to_dict()
    assert d["spec"]["encoding"]["x"]["title"] == "Meu eixo"


# ---------------------------------------------------------------- boxplot


def test_boxplot_um_por_grupo_e_dominio_comum(pop_pequena):
    d = boxplot(pop_pequena, "spread", (0, 6), "t").to_dict()
    dominios = [v.get("scale", {}).get("domain") for v in _encodings(d)]
    assert [0, 6] in dominios
    caixas = next(t for t in _tabelas(d) if "q1" in t[0])
    assert {r["grupo"] for r in caixas} == {"N+NE", "Demais"}
    assert len(caixas) == 2
    demais = next(r for r in caixas if r["grupo"] == "Demais")
    valores = pop_pequena.loc[pop_pequena["grupo"] == "Demais", "spread"].dropna()
    assert demais["q1"] == pytest.approx(valores.quantile(0.25))
    assert demais["mediana"] == pytest.approx(valores.median())
    assert demais["q3"] == pytest.approx(valores.quantile(0.75))


def test_boxplot_bigodes_param_dentro_dos_dados_e_pontos_fora(pop_pequena):
    d = boxplot(pop_pequena, "valor_contratado", (0, 1e8), "t").to_dict()
    caixas = next(t for t in _tabelas(d) if "q1" in t[0])
    for linha in caixas:
        assert linha["inferior"] <= linha["q1"] <= linha["mediana"] <= linha["q3"] <= linha["superior"]
    pontos = [t for t in _tabelas(d) if t and "q1" not in t[0]]
    fora = pd.DataFrame(pontos[0]) if pontos else pd.DataFrame(columns=["valor"])
    # tudo o que passa de 1,5 amplitude interquartil dos extremos aparece como ponto
    esperado = 0
    for _, valores in pop_pequena.groupby("grupo")["valor_contratado"]:
        q1, q3 = valores.quantile([0.25, 0.75])
        esperado += int(((valores < q1 - 1.5 * (q3 - q1)) | (valores > q3 + 1.5 * (q3 - q1))).sum())
    assert len(fora) == esperado


def test_boxplot_cores_fixas(pop_pequena):
    d = boxplot(pop_pequena, "prazo_amortizacao", (0, 120), "t").to_dict()
    escalas = _cores_do_grupo(d)
    assert escalas
    for escala in escalas:
        assert escala == {"domain": ["N+NE", "Demais"], "range": [CORES["N+NE"], CORES["Demais"]]}


# ---------------------------------------------------------------- qqplot


def test_qqplot_descarta_nan_e_traz_reta_de_referencia(pop_pequena):
    d = qqplot(pop_pequena["spread"], "t").to_dict()
    pontos = next(t for t in _tabelas(d) if len(t) > 2)
    assert len(pontos) == 17
    teorico = [r["teorico"] for r in pontos]
    assert teorico == sorted(teorico)
    observado = [r["observado"] for r in pontos]
    assert observado == sorted(observado)
    reta = next(t for t in _tabelas(d) if len(t) == 2)
    assert reta[0]["teorico"] < reta[1]["teorico"]
    # as duas camadas aparecem na legenda
    nomes = {c.get("field") for c in _encodings(d)}
    assert "serie" in nomes


def test_qqplot_aceita_array_e_limita_pontos():
    x = np.random.default_rng(1).normal(size=20_000)
    d = qqplot(x, "t").to_dict()  # 20 mil linhas estourariam o limite do Altair
    pontos = max(_tabelas(d), key=len)
    assert 2 < len(pontos) <= 5000
    assert pontos[0]["observado"] == pytest.approx(x.min())
    assert pontos[-1]["observado"] == pytest.approx(x.max())


def test_qqplot_precisa_de_valores():
    with pytest.raises(ValueError, match="valores"):
        qqplot([1.0, float("nan")], "t")


# ---------------------------------------------------------------- intervalos


def _resultados() -> list[dict]:
    return [
        {"id": "H1", "estimativa": 0.55, "li": 0.50, "ls": 0.60, "verdadeiro": 0.52, "unidade": "proporção"},
        {"id": "H2", "estimativa": 1.90, "li": 1.20, "ls": 2.60, "verdadeiro": 2.10, "unidade": "% a.a."},
        {"id": "H3", "estimativa": -14.0, "li": -20.0, "ls": -8.0, "verdadeiro": -11.0, "unidade": "meses"},
        {"id": "H4", "estimativa": 3.2e7, "li": 1.0e7, "ls": 5.4e7, "verdadeiro": 4.0e7, "unidade": "R$"},
        {"id": "H5", "estimativa": 0.10, "li": -0.05, "ls": 0.25, "verdadeiro": 0.12, "unidade": "p.p."},
    ]


def test_intervalos_cinco_itens_com_unidades_diferentes():
    d = intervalos(_resultados(), "t").to_dict()
    assert d["facet"]["row"]["field"] == "rotulo"
    assert d["resolve"]["scale"]["x"] == "independent"
    tabela = _tabelas(d)[0]
    assert [r["id"] for r in tabela] == ["H1", "H2", "H3", "H4", "H5"]
    assert {r["unidade"] for r in tabela} == {"proporção", "% a.a.", "meses", "R$", "p.p."}


def test_intervalos_legenda_explica_ponto_e_losango():
    d = intervalos(_resultados(), "t").to_dict()
    texto = json.dumps(d, ensure_ascii=False)
    assert "ponto = estimativa da amostra" in texto
    assert "losango = valor verdadeiro da população" in texto
    formas = [c for c in _encodings(d) if c.get("field") == "tipo" and "scale" in c]
    assert any(c["scale"].get("range") == ["circle", "diamond"] for c in formas)


def test_intervalos_um_item_so():
    d = intervalos(_resultados()[:1], "t").to_dict()
    assert len(_tabelas(d)[0]) == 1


# ---------------------------------------------------------------- dominio


def test_dominio_acrescenta_dois_por_cento_e_trava_em_zero():
    pop = pd.DataFrame({"x": [10.0, 20.0, 30.0, np.nan]})
    # amplitude 20 -> folga de 0,4 de cada lado; o mínimo é positivo, então o piso não cruza o zero
    assert dominio(pop, "x") == pytest.approx((9.6, 30.4))
    perto_de_zero = pd.DataFrame({"x": [0.1, 50.0]})
    # amplitude 49,9 -> folga de 0,998: o piso 0,1 - 0,998 seria negativo e fica em 0
    assert dominio(perto_de_zero, "x") == pytest.approx((0.0, 50.998))


def test_dominio_com_minimo_negativo_nao_trava():
    pop = pd.DataFrame({"x": [-10.0, 10.0]})
    assert dominio(pop, "x") == pytest.approx((-10.4, 10.4))


def test_dominio_devolve_tupla_de_floats_e_ignora_nan(pop_pequena):
    lo, hi = dominio(pop_pequena, "spread")
    assert isinstance(lo, float) and isinstance(hi, float)
    validos = pop_pequena["spread"].dropna()
    assert lo < validos.min() and hi > validos.max()
    assert lo >= 0


def test_dominio_coluna_constante_ou_vazia():
    lo, hi = dominio(pd.DataFrame({"x": [5.0, 5.0]}), "x")
    assert lo < 5.0 < hi
    with pytest.raises(ValueError, match="valores"):
        dominio(pd.DataFrame({"x": [np.nan]}), "x")


# ---------------------------------------------------------------- todos


def _todos(pop):
    """Um gráfico de cada construtor, para conferir o que vale para todos."""
    return {
        "barras_regiao contratos": barras_regiao(pop, "contratos", "t"),
        "barras_regiao valor": barras_regiao(pop, "valor", "t"),
        "barras_setor": barras_setor(pop, "t"),
        "histograma": histograma(pop, "spread", dominio(pop, "spread"), "t"),
        "boxplot": boxplot(pop, "spread", dominio(pop, "spread"), "t"),
        "qqplot": qqplot(pop["spread"], "t"),
        "intervalos": intervalos(_resultados(), "t"),
    }


def test_todos_os_construtores_tem_titulo_e_fonte(pop_pequena):
    for nome, grafico in _todos(pop_pequena).items():
        d = grafico.to_dict()
        assert d["title"]["text"] == "t", nome
        assert FONTE in json.dumps(d, ensure_ascii=False), nome
        subtitulo = d["title"]["subtitle"]
        assert FONTE in ([subtitulo] if isinstance(subtitulo, str) else subtitulo), nome


def test_todos_os_construtores_tem_legenda(pop_pequena):
    for nome, grafico in _todos(pop_pequena).items():
        d = grafico.to_dict()
        legendas = [
            c
            for c in _encodings(d)
            if c.get("field") in {"grupo", "serie", "tipo"} and c.get("legend", {}) is not None
        ]
        assert legendas, nome


def test_populacao_grande_nao_estoura_o_limite_de_linhas_do_altair(pop_pequena):
    grande = pd.concat([pop_pequena] * 400, ignore_index=True)  # 8.000 contratos
    for nome, grafico in _todos(grande).items():
        grafico.to_dict()  # MaxRowsError se algum gráfico embutir os 8.000 registros


# ---------------------------------------------------------------- dados reais


def test_graficos_com_a_populacao_e_a_amostra_reais():
    caminho_pop = PASTA_DADOS / "populacao.parquet"
    caminho_amostra = PASTA_DADOS / "amostra.parquet"
    if not (caminho_pop.exists() and caminho_amostra.exists()):
        pytest.skip(f"dados ausentes em {PASTA_DADOS}")
    pop = pd.read_parquet(caminho_pop)
    amostra = pd.read_parquet(caminho_amostra)
    for coluna in ("spread", "prazo_amortizacao"):
        faixa = dominio(pop, coluna)
        for base in (pop, amostra):
            d = histograma(base, coluna, faixa, "t").to_dict()
            assert d["spec"]["encoding"]["x"]["scale"]["domain"] == list(faixa)
            boxplot(base, coluna, faixa, "t").to_dict()
    for base in (pop, amostra):
        barras_regiao(base, "contratos", "t").to_dict()
        barras_regiao(base, "valor", "t").to_dict()
        barras_setor(base, "t").to_dict()
        qqplot(base["spread"], "t").to_dict()
