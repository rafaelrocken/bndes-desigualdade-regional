from calculo import UFS_REGIAO, ler_csv_bndes


def test_le_base_inteira(csv_bruto):
    df = ler_csv_bndes(csv_bruto)
    assert df.shape == (24035, 35)          # 34 colunas + ano
    assert abs(df.valor_contratado_reais.sum() / 1e9 - 1240.2) < 0.1
    assert df.custo_financeiro.str.strip().eq(df.custo_financeiro).all()
    assert df.juros.dtype.kind == "f"


def test_ufs_regiao():
    assert len(UFS_REGIAO) == 27 and UFS_REGIAO["BA"] == "NE" and UFS_REGIAO["AM"] == "N"
