"""Testes das peças de tela no estilo da IA001 (Task 7c): as notas coloridas dos cartões, o
HTML dos cartões e o título que diz a conclusão do primeiro bloco do painel descritivo."""
import pytest

import calculo as c
import graficos as g
import tela
from dados_app import carregar, carregar_ibge

P0_PIB, P0_POP = (h["p0_ou_delta0"] for h in c.HIPOTESES[:2])


# ---------------------------------------------------------------- regra de cor

@pytest.mark.parametrize(
    "dif, sentido, igual, esperado",
    [
        (+0.05, +1, 0.0, "bom"),  # acima da régua: favorece a promessa
        (-0.05, +1, 0.0, "crit"),  # abaixo da régua por mais de 1 p.p.
        (-0.005, +1, 0.0, "neutro"),  # abaixo, mas por menos de 1 p.p.
        (-0.01, +1, 0.0, "neutro"),  # exatamente 1 p.p.: ainda não passa do limiar
        (-0.3, -1, 0.1, "bom"),  # spread menor que o das demais: favorece
        (+0.05, -1, 0.1, "neutro"),  # dentro da faixa "praticamente igual"
        (+0.005, +1, 0.01, "neutro"),  # a favor, mas dentro de 1 p.p.: a zona cinza vale para os dois lados
        (+0.02, +1, 0.01, "bom"),  # a favor e fora da faixa
    ],
)
def test_tom_da_nota(dif, sentido, igual, esperado):
    assert tela.tom_nota(dif, sentido, igual) == esperado


def test_nota_da_regua_acima_e_abaixo():
    abaixo = tela.nota_regua(0.19051, P0_POP, "da população")
    assert abaixo == {"texto": "16,3 p.p. abaixo da régua da população (35,4%)", "tom": "crit", "sobe": False}
    acima = tela.nota_regua(0.29, P0_PIB, "do PIB")
    assert acima == {"texto": "9,4 p.p. acima da régua do PIB (19,6%)", "tom": "bom", "sobe": True}
    perto = tela.nota_regua(0.19051, P0_PIB, "do PIB")
    assert perto["texto"] == "0,5 p.p. abaixo da régua do PIB (19,6%)" and perto["tom"] == "neutro"
    # a favor da promessa, mas dentro de 1 p.p.: também cinza, sem seta
    perto_acima = tela.nota_regua(0.1972, P0_PIB, "do PIB")
    assert perto_acima["texto"] == "0,1 p.p. acima da régua do PIB (19,6%)" and perto_acima["tom"] == "neutro"
    assert "▲" not in tela.nota_html(perto_acima)


def test_nota_das_condicoes():
    igual = tela.nota_condicao(-0.07, 2.72, -1, "um spread menor", tela.QUASE_IGUAL * 2.72)
    assert igual["tom"] == "neutro" and igual["texto"].startswith("praticamente igual")
    # o limiar do spread é relativo à média: 0,9 p.p. a mais (um terço da média) já é vermelho
    caro = tela.nota_condicao(0.9, 2.72, -1, "um spread menor", tela.QUASE_IGUAL * 2.72)
    assert caro["tom"] == "crit" and "a promessa era um spread menor" in caro["texto"]
    prazo = tela.nota_condicao(28.4, 86.6, +1, "um prazo maior", 0.05 * 86.6)
    assert prazo["tom"] == "bom" and prazo["sobe"] and prazo["texto"].startswith("maior em Norte e Nordeste")
    contra = tela.nota_condicao(-20.0, 86.6, +1, "um prazo maior", 0.05 * 86.6)
    assert contra["tom"] == "crit" and "a promessa era um prazo maior" in contra["texto"]


def test_seta_so_nas_notas_coloridas():
    assert "▼" in tela.nota_html({"texto": "x", "tom": "crit", "sobe": False})
    assert "▲" in tela.nota_html({"texto": "x", "tom": "bom", "sobe": True})
    cinza = tela.nota_html({"texto": "x", "tom": "neutro", "sobe": True})
    assert "▲" not in cinza and "▼" not in cinza and 'class="k-nota neutro"' in cinza


# ---------------------------------------------------------------- cartões

def test_cartao_guarda_a_ajuda_e_escapa_o_texto():
    html = tela.cartao_html({"rotulo": "A < B", "valor": "1", "ajuda": 'Diz "isto"'})
    assert "A &lt; B" in html
    assert 'title="Diz &quot;isto&quot;"' in html  # a ajuda continua acessível, no `title`


def test_html_inline_troca_negrito_e_cifrao():
    assert tela.html_inline(r"um contrato de R\$ 1 mil, **grande**") == \
        "um contrato de R&#36; 1 mil, <strong>grande</strong>"


def test_cartoes_do_painel_1_com_as_notas_do_enunciado():
    pop, amostra, _ = carregar()
    contratos, valor, spread, prazo = tela.itens_descritivos(pop, amostra, "populacao")
    assert contratos["valor"] == "19,1%"
    assert [(n["texto"], n["tom"]) for n in contratos["notas"]] == [
        ("16,3 p.p. abaixo da régua da população (35,4%)", "crit"),
        ("0,5 p.p. abaixo da régua do PIB (19,6%)", "neutro"),
    ]
    # o medidor marca as duas réguas, lidas do pré-registro
    assert [m for m, _ in contratos["medidor"]["marcas"]] == [P0_PIB, P0_POP]
    assert [n["tom"] for n in valor["notas"]] == ["crit", "bom"]
    assert spread["notas"][0]["tom"] == "neutro" and spread["notas"][0]["texto"].startswith("praticamente igual")
    assert "médias: 2,66 contra 2,72 % a.a." in spread["sub"]
    assert prazo["notas"][0]["tom"] == "bom"
    assert all(not item["sub"] or "da população" not in item["sub"][0] for item in (contratos, valor))


def test_cartoes_do_painel_2_dizem_a_diferenca_para_a_populacao():
    pop, amostra, _ = carregar()
    itens = tela.itens_descritivos(pop, amostra, "amostra")
    assert itens[0]["valor"] == tela.pct(float((amostra["grupo"] == "N+NE").mean()), 1)
    assert all(item["sub"][0].endswith("da população") for item in itens)
    # 19,7% na amostra contra a régua do PIB (19,6%): ruído de sorteio, cinza e sem seta
    nota_pib = itens[0]["notas"][1]
    assert nota_pib["texto"] == "0,1 p.p. acima da régua do PIB (19,6%)" and nota_pib["tom"] == "neutro"
    assert "▲" not in tela.nota_html(nota_pib)


# ---------------------------------------------------------------- título do bloco

@pytest.mark.parametrize("nivel", ["regiao", "estado"])
def test_titulo_do_primeiro_bloco_vem_dos_dados(nivel):
    pop, _, _ = carregar()
    unid = g.unidades(carregar_ibge(), nivel)
    resumo = g.por_unidade(pop, unid, nivel)
    titulo = tela.titulo_onde(resumo, unid, nivel)
    coluna = "regiao" if nivel == "regiao" else "uf"
    partes = pop[coluna].value_counts(normalize=True)
    nne = float((pop["grupo"] == "N+NE").mean())
    # a maior parte dos contratos, contada direto na população, e a do Norte e Nordeste
    assert tela.pct(partes.max()) in titulo and tela.pct(nne) in titulo
    if nivel == "regiao":
        assert titulo == "O **Sudeste** concentra **46,7% dos contratos**; Norte e Nordeste juntos ficam com **19,1%**."
    else:
        assert titulo.startswith("Só **São Paulo (SP)** tem") and "os 16 estados do Norte e do Nordeste" in titulo
