import numpy as np
import pandas as pd
import pytest
from scipy import stats

from calculo import (
    HIPOTESES,
    INDEXADORES_REAIS,
    UFS_REGIAO,
    agregar_contratos,
    condicoes_media,
    condicoes_proporcao,
    descritiva,
    filtrar_subcreditos,
    ic_media,
    julgar_decisao,
    ler_csv_bndes,
    montar_populacao,
    poder_welch,
    poder_z_proporcao,
    sortear_amostra,
    tamanho_amostra,
    teste_welch,
    teste_z_proporcao,
)


def _sub(linhas):
    """Monta subcréditos sintéticos com as colunas que `agregar_contratos` usa.

    Cada linha é (numero_do_contrato, uf, valor, custo_financeiro, juros,
    prazo_amortizacao). O resto recebe valores-padrão.
    """
    sub = pd.DataFrame(
        linhas,
        columns=[
            "numero_do_contrato",
            "uf",
            "valor_contratado_reais",
            "custo_financeiro",
            "juros",
            "prazo_amortizacao_meses",
        ],
    )
    sub["porte_do_cliente"] = "GRANDE"
    sub["setor_bndes"] = "INDUSTRIA"
    sub["data_da_contratacao"] = pd.Timestamp("2015-01-01")
    sub["ano"] = 2015
    sub["cnpj"] = "0"
    sub["modalidade_de_apoio"] = "REEMBOLSÁVEL"
    return sub


def test_le_base_inteira(csv_bruto):
    df = ler_csv_bndes(csv_bruto)
    assert df.shape == (24035, 35)          # 34 colunas + ano
    assert abs(df.valor_contratado_reais.sum() / 1e9 - 1240.2) < 0.1
    assert df.custo_financeiro.str.strip().eq(df.custo_financeiro).all()
    assert df.juros.dtype.kind == "f"


def test_ufs_regiao():
    assert len(UFS_REGIAO) == 27 and UFS_REGIAO["BA"] == "NE" and UFS_REGIAO["AM"] == "N"


def test_indexadores_reais_sao_dez():
    assert len(INDEXADORES_REAIS) == 10


def test_unica_uf_fora_do_mapa_e_interestadual(csv_bruto):
    df = ler_csv_bndes(csv_bruto)
    assert set(df.uf) - set(UFS_REGIAO) == {"IE"}


def test_agrega_regra_de_uf_spread_e_prazo():
    sub = _sub([("A", "BA", 100, "TJLP", 2.0, 60), ("A", "SP", 300, "US$ / CESTA", 1.0, 120),
                ("B", "PE", 50, "TAXA FIXA", 9.0, 24)])
    ct = agregar_contratos(sub).set_index("numero_do_contrato")
    assert ct.loc["A", "uf"] == "SP" and ct.loc["A", "grupo"] == "Demais"
    assert ct.loc["A", "spread"] == 2.0
    assert ct.loc["A", "prazo_amortizacao"] == 105.0
    assert np.isnan(ct.loc["B", "spread"]) and ct.loc["B", "grupo"] == "N+NE"


def test_agregado_tem_colunas_e_uma_linha_por_contrato():
    sub = _sub([("A", "BA", 100, "TJLP", 2.0, 60), ("A", "SP", 300, "IPCA", 4.0, 120),
                ("B", "PE", 50, "TAXA FIXA", 9.0, 24)])
    ct = agregar_contratos(sub)
    assert list(ct.columns) == [
        "numero_do_contrato", "uf", "regiao", "grupo", "valor_contratado", "spread",
        "prazo_amortizacao", "porte", "setor", "data", "ano", "cnpj", "n_subcreditos",
    ]
    assert len(ct) == 2 and ct.numero_do_contrato.is_unique
    a = ct.set_index("numero_do_contrato").loc["A"]
    assert a.valor_contratado == 400 and a.n_subcreditos == 2
    assert a.regiao == "SE"
    # spread ponderado pelo valor entre os subcréditos em reais: (100*2 + 300*4) / 400
    assert a.spread == 3.5


def test_empate_de_uf_fica_com_a_primeira_na_ordem_original():
    sub = _sub([("A", "BA", 100, "TJLP", 2.0, 60), ("A", "SP", 100, "TJLP", 2.0, 60),
                ("B", "SP", 100, "TJLP", 2.0, 60), ("B", "BA", 100, "TJLP", 2.0, 60)])
    ct = agregar_contratos(sub).set_index("numero_do_contrato")
    assert ct.loc["A", "uf"] == "BA" and ct.loc["A", "grupo"] == "N+NE"
    assert ct.loc["B", "uf"] == "SP" and ct.loc["B", "grupo"] == "Demais"


def test_atributos_do_maior_subcredito_e_desempate():
    sub = _sub([("A", "BA", 100, "TJLP", 2.0, 60), ("A", "SP", 300, "TJLP", 2.0, 60),
                ("B", "BA", 100, "TJLP", 2.0, 60), ("B", "BA", 100, "TJLP", 2.0, 60)])
    sub["porte_do_cliente"] = ["MICRO", "GRANDE", "MEDIA", "PEQUENA"]
    sub["setor_bndes"] = ["AGRO", "INDUSTRIA", "COMERCIO", "INFRA"]
    sub["cnpj"] = ["1", "2", "3", "4"]
    sub["ano"] = [2010, 2011, 2012, 2013]
    sub["data_da_contratacao"] = pd.to_datetime(["2010-05-01", "2011-05-01", "2012-05-01", "2013-05-01"])
    ct = agregar_contratos(sub).set_index("numero_do_contrato")
    a, b = ct.loc["A"], ct.loc["B"]
    assert (a.porte, a.setor, a.cnpj, a.ano) == ("GRANDE", "INDUSTRIA", "2", 2011)
    assert a.data == pd.Timestamp("2011-05-01")
    # empate no maior valor: fica o primeiro na ordem original
    assert (b.porte, b.setor, b.cnpj, b.ano) == ("MEDIA", "COMERCIO", "3", 2012)


def test_contrato_de_valor_zero_nao_quebra_e_fica_com_media_nan():
    sub = _sub([("A", "BA", 0, "TJLP", 2.0, 60), ("A", "SP", 0, "TJLP", 4.0, 120)])
    ct = agregar_contratos(sub)
    assert len(ct) == 1
    assert np.isnan(ct.spread.iloc[0]) and np.isnan(ct.prazo_amortizacao.iloc[0])  # peso total zero


def test_filtra_reembolsavel_uf_valida_e_ano():
    sub = _sub([
        ("A", "BA", 10, "TJLP", 1.0, 12),       # fica
        ("B", "IE", 10, "TJLP", 1.0, 12),       # interestadual: sai
        ("C", "SP", 10, "TJLP", 1.0, 12),       # modalidade não reembolsável: sai
        ("D", "SP", 10, "TJLP", 1.0, 12),       # ano 2026: sai
        ("E", "SP", 10, "TJLP", 1.0, 12),       # ano 2001: sai
        ("F", "SP", 10, "TJLP", 1.0, 12),       # ano 2002: fica
        ("G", "SP", 10, "TJLP", 1.0, 12),       # ano 2025: fica
    ])
    sub["modalidade_de_apoio"] = ["REEMBOLSÁVEL", "REEMBOLSÁVEL", "NÃO REEMBOLSÁVEL",
                                  "REEMBOLSÁVEL", "REEMBOLSÁVEL", "REEMBOLSÁVEL", "REEMBOLSÁVEL"]
    sub["ano"] = [2015, 2015, 2015, 2026, 2001, 2002, 2025]
    assert list(filtrar_subcreditos(sub).numero_do_contrato) == ["A", "F", "G"]


def test_populacao_real(csv_bruto):
    pop = montar_populacao(csv_bruto)
    assert len(pop) == 6320 and pop.numero_do_contrato.is_unique
    assert 0.185 < (pop.grupo == "N+NE").mean() < 0.195


# --- Amostra e inferência (tarefa 3) -----------------------------------------------


def test_tamanho():
    r = tamanho_amostra(6320)
    assert r["n"] == 913 and abs(r["n0"] - 1067.07) < 0.05


def test_tamanho_chaves_z_e_outros_parametros():
    r = tamanho_amostra(6320)
    assert set(r) == {"z", "n0", "n"}
    assert abs(r["z"] - 1.959963984540054) < 1e-12
    assert isinstance(r["n"], int)
    # população menor, mesma margem: n0 = 1067,07; n = 1067,07 / (1 + 1067,07/1000) = 516,22 -> 517
    assert tamanho_amostra(1000)["n"] == 517
    # margem de 5 pontos, 90%: z = 1,6449; n0 = 1,6449^2 * 0,25 / 0,05^2 = 270,55;
    # n = 270,55 / (1 + 270,55/6320) = 259,4 -> 260
    r2 = tamanho_amostra(6320, E=0.05, confianca=0.90)
    assert abs(r2["z"] - 1.6448536269514722) < 1e-12 and r2["n"] == 260


def test_sorteio_reprodutivel():
    pop = pd.DataFrame({"numero_do_contrato": range(6320)})
    a, b = sortear_amostra(pop), sortear_amostra(pop.sample(frac=1, random_state=1))
    assert len(a) == 913 and a.numero_do_contrato.tolist() == b.numero_do_contrato.tolist()


def test_sorteio_sem_reposicao_igual_ao_desenho_e_semente_muda_o_resultado():
    pop = pd.DataFrame({"numero_do_contrato": [f"C{i:05d}" for i in range(6320)], "x": range(6320)})
    a = sortear_amostra(pop)
    assert a.numero_do_contrato.is_unique and set(a.numero_do_contrato) <= set(pop.numero_do_contrato)
    esperado = pop.sort_values("numero_do_contrato").sample(n=913, replace=False, random_state=20261112)
    assert a.equals(esperado)
    assert list(a.columns) == ["numero_do_contrato", "x"]
    outra = sortear_amostra(pop, n=50, semente=1)
    assert len(outra) == 50 and outra.numero_do_contrato.tolist() != a.numero_do_contrato.tolist()[:50]


def test_descritiva_ignora_nan_e_usa_ddof_1():
    s = pd.Series([1.0, 2, 3, 4, np.nan, 10])
    d = descritiva(s)
    assert list(d.index) == ["n", "media", "mediana", "desvio_padrao", "q1", "q3", "minimo", "maximo"]
    assert d["n"] == 5 and d["media"] == 4.0 and d["mediana"] == 3.0
    assert abs(d["desvio_padrao"] - np.sqrt(12.5)) < 1e-12      # sum((x-4)^2) = 50; 50/4 = 12,5
    assert d["q1"] == 2.0 and d["q3"] == 4.0 and d["minimo"] == 1.0 and d["maximo"] == 10.0


def test_ic_media_bate_com_t_de_student():
    x = np.array([2.0, 4, 4, 4, 5, 5, 7, 9, np.nan])
    m, li, ls = ic_media(x)
    v = x[~np.isnan(x)]
    assert abs(m - v.mean()) < 1e-12
    ref = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / np.sqrt(len(v)))
    assert abs(li - ref[0]) < 1e-12 and abs(ls - ref[1]) < 1e-12
    m99, li99, ls99 = ic_media(pd.Series(x), confianca=0.99)
    assert li99 < li < m < ls < ls99


def test_z_proporcao():
    r = teste_z_proporcao(60, 100, 0.5)
    assert abs(r["z"] - 2.0) < 1e-9 and abs(r["p_valor"] - 0.0455) < 1e-3 and r["rejeita"]
    assert np.allclose(r["ic_wald"], (0.5040, 0.6960), atol=1e-3)
    assert np.allclose(r["ic_wilson"], (0.5020, 0.6906), atol=1e-3)


def test_z_proporcao_ep_usa_p0_e_nao_rejeita_perto_de_p0():
    r = teste_z_proporcao(52, 100, 0.4)
    assert set(r) == {"p_hat", "ep_teste", "z", "p_valor", "ic_wald", "ic_wilson", "rejeita"}
    assert r["p_hat"] == 0.52 and abs(r["ep_teste"] - np.sqrt(0.4 * 0.6 / 100)) < 1e-15
    assert abs(r["z"] - 0.12 / np.sqrt(0.0024)) < 1e-12
    r2 = teste_z_proporcao(51, 100, 0.5)
    assert not r2["rejeita"] and abs(r2["p_valor"] - 2 * stats.norm.sf(0.2)) < 1e-12
    assert isinstance(r2["ic_wald"], tuple) and isinstance(r2["ic_wilson"], tuple)
    # bilateral e simétrico: x e n - x em torno de p0 = 0,5 dão o mesmo p-valor
    assert abs(teste_z_proporcao(40, 100, 0.5)["p_valor"] - teste_z_proporcao(60, 100, 0.5)["p_valor"]) < 1e-12
    # alfa controla só a decisão (p = 0,0455 não é menor que 0,01)
    assert teste_z_proporcao(60, 100, 0.5, alfa=0.01)["rejeita"] is False


def test_welch_bate_com_scipy():
    a = np.array([1.0, 2, 3, 4, 5, np.nan]); b = np.array([2.0, 4, 6, 8, 10, 12])
    r = teste_welch(a, b); ref = stats.ttest_ind(a[~np.isnan(a)], b, equal_var=False)
    assert abs(r["p_valor"] - ref.pvalue) < 1e-12 and r["n_a"] == 5
    r2 = teste_welch(a, b, "menor")
    assert abs(r2["p_valor"] - stats.ttest_ind(a[:5], b, equal_var=False, alternative="less").pvalue) < 1e-12


def test_welch_campos_ic_bilateral_e_d_de_cohen():
    a = np.array([1.0, 2, 3, 4, 5, np.nan]); b = np.array([2.0, np.nan, 4, 6, 8, 10, 12])
    r = teste_welch(a, b)
    assert set(r) == {"n_a", "n_b", "media_a", "media_b", "diferenca", "t", "gl", "p_valor",
                      "ic_diferenca", "d_cohen", "rejeita"}
    assert (r["n_a"], r["n_b"]) == (5, 6) and r["media_a"] == 3.0 and r["media_b"] == 7.0
    assert r["diferenca"] == -4.0
    assert abs(r["t"] - (-2.3763541031440183)) < 1e-12 and abs(r["gl"] - 6.972255729794934) < 1e-12
    # IC bilateral de 95% da diferença: dif -+ t_crit * ep, com o t crítico nos gl de Welch
    ep = np.sqrt(2.5 / 5 + 14 / 6)
    tc = stats.t.ppf(0.975, r["gl"])
    assert abs(r["ic_diferenca"][0] - (-4 - tc * ep)) < 1e-9 and abs(r["ic_diferenca"][1] - (-4 + tc * ep)) < 1e-9
    # d de Cohen com desvio combinado: sqrt((4*2,5 + 5*14)/9) = sqrt(80/9)
    assert abs(r["d_cohen"] - (-4 / np.sqrt(80 / 9))) < 1e-12
    assert r["rejeita"] is True       # p = 0,0493 < 0,05


def test_welch_menor_mantem_ic_bilateral_e_valida_alternativa():
    a = np.array([1.0, 2, 3, 4, 5]); b = np.array([2.0, 4, 6, 8, 10, 12])
    bil, men = teste_welch(a, b), teste_welch(a, b, "menor")
    assert men["ic_diferenca"] == bil["ic_diferenca"]
    assert men["t"] == bil["t"] and men["gl"] == bil["gl"] and men["d_cohen"] == bil["d_cohen"]
    assert abs(men["p_valor"] - bil["p_valor"] / 2) < 1e-12 and men["rejeita"] is True
    assert teste_welch(a, b, alfa=0.04)["rejeita"] is False
    with pytest.raises(ValueError):
        teste_welch(a, b, "maior")
    with pytest.raises(ValueError):
        teste_welch(a, b, "less")


def test_condicoes_grupo_pequeno():
    assert not condicoes_media(29, 500)["ok"] and condicoes_media(30, 30)["ok"]


def test_condicoes_proporcao():
    r = condicoes_proporcao(913, 0.196)
    assert abs(r["np0"] - 913 * 0.196) < 1e-9 and abs(r["n_1_menos_p0"] - 913 * 0.804) < 1e-9 and r["ok"]
    assert not condicoes_proporcao(40, 0.196)["ok"]       # 40 * 0,196 = 7,84 < 10
    assert condicoes_proporcao(50, 0.2)["ok"]             # 10 e 40: exatamente no limite


def test_poder_no_h0_e_alfa():
    assert abs(poder_z_proporcao(0.196, 0.196, 913) - 0.05) < 1e-6
    assert 0.05 < poder_z_proporcao(0.1905, 0.196, 913) < 0.08


def test_poder_z_proporcao_cresce_com_n_e_com_a_distancia():
    assert poder_z_proporcao(0.354, 0.196, 913) > 0.999
    assert poder_z_proporcao(0.25, 0.196, 913) > poder_z_proporcao(0.22, 0.196, 913) > 0.05
    assert poder_z_proporcao(0.22, 0.196, 2000) > poder_z_proporcao(0.22, 0.196, 913)
    assert abs(poder_z_proporcao(0.5, 0.5, 100, alfa=0.01) - 0.01) < 1e-9


def test_poder_welch_e_alfa_no_h0_e_cresce_com_o_efeito():
    assert abs(poder_welch(0.0, 2.0, 3.0, 175, 738) - 0.05) < 1e-9
    assert abs(poder_welch(0.0, 2.0, 3.0, 175, 738, alternativa="menor") - 0.05) < 1e-9
    assert abs(poder_welch(0.0, 2.0, 3.0, 175, 738, alfa=0.01) - 0.01) < 1e-9
    # n = 100 por grupo, dp = 2: ep = 0,2828; delta/ep = -3,5355
    # bilateral: Phi(-1,96 + 3,5355) + (1 - Phi(1,96 + 3,5355)) = 0,9424
    assert abs(poder_welch(-1.0, 2.0, 2.0, 100, 100) - 0.9424) < 1e-3
    # "menor": Phi(-1,645 + 3,5355) = 0,9707, e maior que o bilateral
    p_menor = poder_welch(-1.0, 2.0, 2.0, 100, 100, alternativa="menor")
    assert abs(p_menor - 0.9707) < 1e-3 and p_menor > 0.5
    assert p_menor > poder_welch(-1.0, 2.0, 2.0, 100, 100)
    # diferença no sentido contrário ao da alternativa: poder abaixo de alfa
    assert poder_welch(1.0, 2.0, 2.0, 100, 100, alternativa="menor") < 0.05
    with pytest.raises(ValueError):
        poder_welch(-1.0, 2.0, 2.0, 100, 100, alternativa="maior")


def test_julgar():
    assert julgar_decisao(False, 0.1905, 0.196, "bilateral") == "erro tipo II"
    assert julgar_decisao(True, 0.19, 0.354, "bilateral") == "acerto"
    assert julgar_decisao(True, 0.3, 0.0, "menor") == "erro tipo I"


def test_julgar_todas_as_combinacoes():
    # bilateral: H0 verdadeira só quando o valor verdadeiro é igual ao de H0
    assert julgar_decisao(True, 0.196, 0.196, "bilateral") == "erro tipo I"
    assert julgar_decisao(False, 0.196, 0.196, "bilateral") == "acerto"
    assert julgar_decisao(False, 0.19, 0.354, "bilateral") == "erro tipo II"
    # menor: H0 (valor >= valor_h0) verdadeira em 0,3 e em 0,0; falsa em -0,3
    assert julgar_decisao(False, 0.3, 0.0, "menor") == "acerto"
    assert julgar_decisao(True, 0.0, 0.0, "menor") == "erro tipo I"
    assert julgar_decisao(True, -0.3, 0.0, "menor") == "acerto"
    assert julgar_decisao(False, -0.3, 0.0, "menor") == "erro tipo II"
    with pytest.raises(ValueError):
        julgar_decisao(True, 0.3, 0.0, "maior")


def test_hipoteses_gravadas_como_no_desenho():
    assert [h["id"] for h in HIPOTESES] == ["H1", "H2", "H3", "H4", "H5"]
    assert [h["alternativa"] for h in HIPOTESES] == ["bilateral", "bilateral", "bilateral", "menor", "bilateral"]
    assert [h["p0_ou_delta0"] for h in HIPOTESES] == [0.196, 0.354, 0.0, 0.0, 0.0]
    assert [h["variavel"] for h in HIPOTESES] == ["grupo", "grupo", "spread", "spread", "prazo_amortizacao"]
    assert [h["teste"] for h in HIPOTESES] == ["Z de uma proporção"] * 2 + ["t de Welch"] * 3
    chaves = {"id", "parametro", "h0", "h1", "p0_ou_delta0", "alternativa", "referencia", "variavel", "teste"}
    assert all(set(h) == chaves for h in HIPOTESES)
    assert all(isinstance(h[k], str) and h[k] for h in HIPOTESES for k in ("parametro", "h0", "h1", "referencia"))
