import numpy as np
import pandas as pd

from calculo import (
    INDEXADORES_REAIS,
    UFS_REGIAO,
    agregar_contratos,
    filtrar_subcreditos,
    ler_csv_bndes,
    montar_populacao,
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
