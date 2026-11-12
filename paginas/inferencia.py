"""Painel 3: inferência com a amostra (H1 a H5) e comparação com a população."""
import sys
from pathlib import Path

# A página também roda sozinha (testes, `streamlit run paginas/inferencia.py`): põe a raiz do app no caminho.
_RAIZ = str(Path(__file__).resolve().parents[1])
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

import math  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy import stats  # noqa: E402

import calculo as c  # noqa: E402
import graficos as g  # noqa: E402
from dados_app import carregar, carregar_simulacao  # noqa: E402
from tela import GRUPOS, br, faixa, fp, grafico, inteiro, pct, tabela, tabela_estatica  # noqa: E402

ALFA = 0.05  # fixado antes do sorteio
HIP = {h["id"]: h for h in c.HIPOTESES}
UNIDADE = {"grupo": "% dos contratos", "spread": "p.p. de spread", "prazo_amortizacao": "meses"}
NOME_VAR = {"spread": "Spread (% a.a.)", "prazo_amortizacao": "Prazo de amortização (meses)"}

pop, amostra, meta = carregar()
n = len(amostra)
x = int((amostra["grupo"] == "N+NE").sum())
pi_pop = float((pop["grupo"] == "N+NE").mean())
valor_nne = float(pop.loc[pop["grupo"] == "N+NE", "valor_contratado"].sum() / pop["valor_contratado"].sum())


def formatar(var: str, v: float) -> str:
    """Valor na unidade da variável: proporção em %, spread em p.p., prazo em meses."""
    if var == "grupo":
        return pct(v, 2)
    if var == "spread":
        return br(v, 3) + " p.p."
    return br(v, 2) + " meses"


def decisao(rejeita: bool) -> str:
    return "rejeita H0" if rejeita else "não rejeita H0"


# --- Os cinco testes (só a amostra) e a verdade da população ---------------------------
amostra_grupo = {
    var: {gr: amostra.loc[amostra["grupo"] == gr, var].dropna() for gr in GRUPOS} for var in NOME_VAR
}
pop_grupo = {
    var: {gr: c.descritiva(pop.loc[pop["grupo"] == gr, var]) for gr in GRUPOS} for var in NOME_VAR
}
delta_pop = {var: float(pop_grupo[var]["N+NE"]["media"] - pop_grupo[var]["Demais"]["media"]) for var in NOME_VAR}
d_pop = {}
for var in NOME_VAR:
    da, db = pop_grupo[var]["N+NE"], pop_grupo[var]["Demais"]
    dp_comb = math.sqrt(((da["n"] - 1) * da["desvio_padrao"] ** 2 + (db["n"] - 1) * db["desvio_padrao"] ** 2)
                        / (da["n"] + db["n"] - 2))
    d_pop[var] = delta_pop[var] / dp_comb

R = {}  # resultados por hipótese
for h in c.HIPOTESES:
    hid, var, alt, v0 = h["id"], h["variavel"], h["alternativa"], h["p0_ou_delta0"]
    if var == "grupo":
        r = c.teste_z_proporcao(x, n, v0, ALFA)
        R[hid] = {
            "teste": r, "estimativa": r["p_hat"], "ic": r["ic_wald"], "verdadeiro": pi_pop,
            "condicoes": c.condicoes_proporcao(n, v0), "poder": c.poder_z_proporcao(pi_pop, v0, n, ALFA),
        }
    else:
        a, b = amostra_grupo[var]["N+NE"], amostra_grupo[var]["Demais"]
        r = c.teste_welch(a, b, alt, ALFA)
        poder = c.poder_welch(delta_pop[var], pop_grupo[var]["N+NE"]["desvio_padrao"],
                              pop_grupo[var]["Demais"]["desvio_padrao"], r["n_a"], r["n_b"], ALFA, alt)
        R[hid] = {
            "teste": r, "estimativa": r["diferenca"], "ic": r["ic_diferenca"], "verdadeiro": delta_pop[var],
            "condicoes": c.condicoes_media(r["n_a"], r["n_b"]), "poder": poder,
        }
        if alt == "menor":
            # IC unilateral de 95% que combina com o teste à esquerda a 5%: (−∞; δ̂ + t(0,95; gl) · ep]
            ep = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            R[hid]["limite_unilateral"] = r["diferenca"] + float(stats.t.ppf(1 - ALFA, r["gl"])) * ep
    resultado = R[hid]
    resultado["contem"] = resultado["ic"][0] <= resultado["verdadeiro"] <= resultado["ic"][1]
    resultado["h0_verdadeira"] = (resultado["verdadeiro"] == v0) if alt == "bilateral" else (resultado["verdadeiro"] >= v0)
    resultado["julgamento"] = c.julgar_decisao(r["rejeita"], resultado["verdadeiro"], v0, alt)


# --- Cabeçalho -------------------------------------------------------------------------
st.title("Painel 3 — Inferência")
st.markdown(
    f"Os testes e os intervalos usam **só a amostra** ({inteiro(n)} contratos). Nível de significância "
    f"**α = {br(ALFA)}**, fixado antes do sorteio, junto com as cinco hipóteses. A população entra "
    "apenas no fim, na comparação: o intervalo contém o valor verdadeiro? A decisão do teste acertou?"
)
st.caption(
    "H1 a H5 são os nomes dos cinco testes. A hipótese alternativa de cada teste aparece escrita por "
    "extenso, para não confundir com o teste H1."
)


# --- Um bloco por hipótese -------------------------------------------------------------
def bloco_proporcao(hid: str) -> None:
    h, res = HIP[hid], R[hid]
    r, cond = res["teste"], res["condicoes"]
    st.markdown(
        f"**{hid}: {h['referencia']}.**  \n"
        f"Parâmetro: {h['parametro']}. Variável qualitativa: o grupo do contrato; evento de interesse: "
        "o contrato ser N+NE."
    )
    st.markdown(f"Hipótese nula: {h['h0']}  \nHipótese alternativa: {h['h1']}")
    st.markdown(f"Teste: **{h['teste']}**, com $p_0$ no erro padrão. α = {br(ALFA)}.")
    st.latex(r"z = \frac{\hat p - p_0}{\sqrt{p_0(1-p_0)/n}}")
    tabela(pd.DataFrame(
        [
            ("Contratos N+NE na amostra (x)", inteiro(x)),
            ("Tamanho da amostra (n)", inteiro(n)),
            ("Estimativa p̂ = x / n", pct(r["p_hat"], 2)),
            ("Erro padrão do teste (com p₀)", pct(r["ep_teste"], 3)),
            ("Estatística z", br(r["z"], 3)),
            ("Valor-p (bilateral)", fp(r["p_valor"])),
            ("IC 95% de Wald para π", faixa(r["ic_wald"], lambda v: pct(v, 2))),
            ("IC 95% de Wilson para π", faixa(r["ic_wilson"], lambda v: pct(v, 2))),
            ("Decisão (α = 0,05)", decisao(r["rejeita"])),
        ],
        columns=["Item", "Valor"],
    ))
    sinal = "<" if r["rejeita"] else "≥"
    st.markdown(
        f"**Decisão: {decisao(r['rejeita'])}** (valor-p {fp(r['p_valor'])} {sinal} {br(ALFA)}). "
        f"O intervalo de 95% {'não contém' if r['rejeita'] else 'contém'} o valor de H0 "
        f"({pct(h['p0_ou_delta0'], 1)}), coerente com a decisão."
    )
    st.markdown(
        f"**Condições:** $n\\,p_0$ = {br(cond['np0'], 1)} e $n\\,(1-p_0)$ = {br(cond['n_1_menos_p0'], 1)}, "
        f"{'ambos' if cond['ok'] else 'nem ambos'} ≥ 10; amostra aleatória simples. "
        f"**Poder do teste** (com o π verdadeiro da população): {pct(res['poder'], 1)}."
    )


def qqplots(var: str) -> None:
    esquerda, direita = st.columns(2)
    for coluna, gr in zip((esquerda, direita), GRUPOS):
        valores = amostra_grupo[var][gr]
        with coluna:
            grafico(g.qqplot(valores, f"QQ-plot do {NOME_VAR[var].split(' (')[0].lower()}: {gr} (n = {inteiro(len(valores))})"))
    st.caption(
        "Os pontos se afastam da reta nas pontas (cauda direita pesada) e andam em degraus (valores "
        "repetidos e zeros): os dados não são normais. Com pelo menos 170 contratos por grupo, o Teorema "
        "Central do Limite sustenta o teste t, porque a média de muitos contratos é aproximadamente normal."
    )


def bloco_media(hid: str) -> None:
    h, res = HIP[hid], R[hid]
    r, cond, var = res["teste"], res["condicoes"], h["variavel"]
    casas = 3 if var == "spread" else 2
    f = (lambda v: br(v, casas))
    st.markdown(f"**{hid}: {h['referencia']}.**  \nParâmetro: {h['parametro']}. Variável: {NOME_VAR[var]}.")
    st.markdown(f"Hipótese nula: {h['h0']}  \nHipótese alternativa: {h['h1']}")
    lado = "unilateral à esquerda" if h["alternativa"] == "menor" else "bilateral"
    st.markdown(f"Teste: **{h['teste']}** ({lado}), sem supor variâncias iguais. α = {br(ALFA)}.")
    st.latex(r"t = \frac{\bar x_{N+NE} - \bar x_{Demais}}{\sqrt{s^2_{N+NE}/n_{N+NE} + s^2_{Demais}/n_{Demais}}}")
    linhas = [
        ("n N+NE / n Demais", f"{inteiro(r['n_a'])} / {inteiro(r['n_b'])}"),
        ("Média N+NE / média Demais", f"{f(r['media_a'])} / {f(r['media_b'])}"),
        ("Estimativa δ̂ (N+NE − Demais)", f(r["diferenca"])),
        ("Estatística t (graus de liberdade de Welch)", f"{br(r['t'], 3)} (gl = {br(r['gl'], 1)})"),
        (f"Valor-p ({lado})", fp(r["p_valor"])),
        ("IC 95% bilateral da diferença", faixa(r["ic_diferenca"], f)),
    ]
    if "limite_unilateral" in res:
        linhas.append(("IC unilateral de 95% (combina com o teste)", f"(−∞; {f(res['limite_unilateral'])}]"))
    for gr in GRUPOS:
        media, inf, sup = c.ic_media(amostra_grupo[var][gr])
        linhas.append((f"IC 95% da média de {gr}", faixa((inf, sup), f)))
    linhas += [("d de Cohen", br(r["d_cohen"], 3)), ("Decisão (α = 0,05)", decisao(r["rejeita"]))]
    tabela(pd.DataFrame(linhas, columns=["Item", "Valor"]))

    sinal = "<" if r["rejeita"] else "≥"
    st.markdown(f"**Decisão: {decisao(r['rejeita'])}** (valor-p {fp(r['p_valor'])} {sinal} {br(ALFA)}).")
    if "limite_unilateral" in res:
        u = res["limite_unilateral"]
        st.markdown(
            f"O IC da tabela, {faixa(r['ic_diferenca'], f)}, é o **bilateral** de 95%: o intervalo para "
            "estimar δ, igual ao de H3. O intervalo que combina com o teste unilateral à esquerda a 5% é "
            f"um limite superior, $(-\\infty;\\,U]$ com $U = \\hat\\delta + t_{{0{{,}}95;\\,gl}} \\cdot ep$ = {f(u)}. "
            f"Rejeitar H0 equivale a U ficar abaixo de zero; aqui U {'é negativo' if u < 0 else 'é positivo e o intervalo contém o 0'}, "
            f"coerente com {decisao(r['rejeita'])}."
        )
    st.markdown(
        f"**Condições:** amostra aleatória simples; n ≥ 30 nos dois grupos: "
        f"{inteiro(cond['n_a'])} e {inteiro(cond['n_b'])} ({'sim' if cond['ok'] else 'não'}); QQ-plots abaixo. "
        f"**Poder do teste** (com a diferença verdadeira da população): {pct(res['poder'], 1)}."
    )
    qqplots(var)


st.header("Os cinco testes")
abas = st.tabs(["H1 · régua do PIB", "H2 · régua da população", "H3 · spread igual?",
                "H4 · N+NE pagam menos?", "H5 · prazo igual?"])
for aba, hid in zip(abas, HIP):
    with aba:
        (bloco_proporcao if HIP[hid]["variavel"] == "grupo" else bloco_media)(hid)


# --- Comparação com a população --------------------------------------------------------
st.header("Comparação com a população: acertamos?")
st.markdown(
    "Na população o valor verdadeiro é conhecido. **Como julgamos acerto:** se o valor verdadeiro é "
    "diferente do de H0, rejeitar é acerto e não rejeitar é **erro tipo II**; se é igual (ou, em H4, se "
    "cai do lado de H0), rejeitar seria **erro tipo I**. Para H3 a H5, o valor verdadeiro é a diferença "
    "de médias da população. O **poder** é a chance de o teste rejeitar H0 numa amostra como a nossa, se "
    "o valor verdadeiro for o da população (aproximação normal)."
)
st.markdown(
    "Dois cuidados. Numa população finita e real, o valor verdadeiro quase nunca é **exatamente** o de "
    "H0, então o erro tipo I aqui é hipotético. E um erro tipo II não quer dizer teste mal feito: quando "
    "a diferença verdadeira é pequena, o poder é baixo e o teste quase sempre deixa de rejeitar."
)

linhas = []
for hid, h in HIP.items():
    res, var = R[hid], h["variavel"]
    r = res["teste"]
    alternativo = "-"
    if var == "grupo":
        w = r["ic_wilson"]
        alternativo = f"Wilson {faixa(w, lambda v: formatar(var, v))}, contém: {'sim' if w[0] <= res['verdadeiro'] <= w[1] else 'não'}"
    elif "limite_unilateral" in res:
        u = res["limite_unilateral"]
        alternativo = f"unilateral (−∞; {formatar(var, u)}], contém: {'sim' if res['verdadeiro'] <= u else 'não'}"
    estatistica = f"z = {br(r['z'], 2)}" if var == "grupo" else f"t = {br(r['t'], 2)} (gl {br(r['gl'], 0)})"
    v0 = h["p0_ou_delta0"]
    linhas.append({
        "Teste": hid,
        "Parâmetro": "π" if var == "grupo" else f"δ do {'spread' if var == 'spread' else 'prazo'}",
        "H0": f"π = {pct(v0, 1)}" if var == "grupo" else ("δ ≥ 0" if h["alternativa"] == "menor" else "δ = 0"),
        "Estimativa (amostra)": formatar(var, res["estimativa"]),
        "IC 95% (bilateral)": faixa(res["ic"], lambda v: formatar(var, v)),
        "Valor verdadeiro (população)": formatar(var, res["verdadeiro"]),
        "O IC contém o verdadeiro?": "sim" if res["contem"] else "não",
        "IC alternativo": alternativo,
        "Estatística": estatistica,
        "Valor-p": fp(r["p_valor"]),
        "Decisão": decisao(r["rejeita"]),
        "H0 é verdadeira na população?": "sim" if res["h0_verdadeira"] else "não",
        "Julgamento": res["julgamento"],
        "Poder": pct(res["poder"], 1),
    })
# Uma coluna por teste e uma linha por item: cinco colunas estreitas cabem na largura da página,
# e o `st.table` quebra o texto longo em vez de cortar.
comparacao = pd.DataFrame(linhas).set_index("Teste").T.rename_axis("Item").reset_index()
tabela_estatica(comparacao)

grafico(g.intervalos(
    [
        {
            "id": hid,
            "estimativa": (100 if HIP[hid]["variavel"] == "grupo" else 1) * R[hid]["estimativa"],
            "li": (100 if HIP[hid]["variavel"] == "grupo" else 1) * R[hid]["ic"][0],
            "ls": (100 if HIP[hid]["variavel"] == "grupo" else 1) * R[hid]["ic"][1],
            "verdadeiro": (100 if HIP[hid]["variavel"] == "grupo" else 1) * R[hid]["verdadeiro"],
            "unidade": UNIDADE[HIP[hid]["variavel"]],
        }
        for hid in HIP
    ],
    "Intervalos de confiança de 95% da amostra e o valor verdadeiro da população",
))
st.caption("H1 e H2 dividem o mesmo intervalo para π; H3 e H4, o mesmo intervalo bilateral para δ do spread.")

st.markdown("**A média de cada grupo: população × amostra** (item b do enunciado: estimar a média populacional)")
linhas = []
for var, nome in NOME_VAR.items():
    for gr in GRUPOS:
        media_pop = float(pop_grupo[var][gr]["media"])
        media, inf, sup = c.ic_media(amostra_grupo[var][gr])
        linhas.append({
            "Variável": nome, "Grupo": gr,
            "Média na população": br(media_pop, 3), "Média na amostra": br(media, 3),
            "Erro amostral": br(media - media_pop, 3),
            "IC 95% da média": faixa((inf, sup), lambda v: br(v, 3)),
            "O IC contém a média da população?": "sim" if inf <= media_pop <= sup else "não",
        })
tabela(pd.DataFrame(linhas))


# --- Interpretação ---------------------------------------------------------------------
def frase_julgamento(res: dict) -> str:
    if res["julgamento"] == "erro tipo II":
        return "**erro tipo II**"
    return f"**{res['julgamento']}**"


st.subheader("Interpretação, teste por teste")
r1, r2, r3, r4, r5 = (R[h] for h in ("H1", "H2", "H3", "H4", "H5"))
p0_1, p0_2 = HIP["H1"]["p0_ou_delta0"], HIP["H2"]["p0_ou_delta0"]
inofensivo = {hid: R[hid]["julgamento"] == "erro tipo II" and R[hid]["poder"] < 0.2 for hid in R}

st.markdown(
    f"**H1 (régua do PIB).** A amostra deu p̂ = {pct(r1['estimativa'], 2)} e o teste "
    f"{'rejeitou' if r1['teste']['rejeita'] else 'não rejeitou'} H0 (p = {fp(r1['teste']['p_valor'])}). "
    f"Na população, N+NE têm {pct(pi_pop, 2)} dos contratos, {br(100 * abs(pi_pop - p0_1), 2)} ponto "
    f"{'abaixo' if pi_pop < p0_1 else 'acima'} da régua de {pct(p0_1, 1)}, então H0 é falsa por pouco e a decisão é "
    f"{'um ' + frase_julgamento(r1) + ' inofensivo' if inofensivo['H1'] else frase_julgamento(r1)}: o poder era de "
    f"só {pct(r1['poder'], 1)}, o teste é quase cego para uma distância tão pequena. Na prática, em número de "
    "contratos, N+NE recebem quase exatamente a parte que lhes caberia pelo PIB, e o IC "
    f"{faixa(r1['ic'], lambda v: pct(v, 2))} {'contém' if r1['contem'] else 'não contém'} o π verdadeiro."
)
st.markdown(
    f"**H2 (régua da população).** O teste {'rejeitou' if r2['teste']['rejeita'] else 'não rejeitou'} H0 "
    f"(z = {br(r2['teste']['z'], 2)}) e a decisão foi {frase_julgamento(r2)}: o π verdadeiro "
    f"({pct(pi_pop, 2)}) está muito longe de {pct(p0_2, 1)}, e o poder era de praticamente "
    f"{pct(r2['poder'], 0)}. Pela régua das pessoas, N+NE recebem bem menos que a sua parte em número de "
    f"contratos. Ressalva da limitação 1: em valor, N+NE ficam com {pct(valor_nne, 0)}, e a régua "
    "populacional é normativa, não uma obrigação do banco."
)
st.markdown(
    f"**H3 (spread, bilateral).** {'Rejeitou' if r3['teste']['rejeita'] else 'Não rejeitou'} H0 "
    f"(p = {fp(r3['teste']['p_valor'])}). Na população, N+NE pagam em média {br(abs(delta_pop['spread']), 3)} "
    f"ponto percentual a {'menos' if delta_pop['spread'] < 0 else 'mais'} de spread, então H0 é falsa e a decisão é "
    f"{'um ' + frase_julgamento(r3) + ' inofensivo' if inofensivo['H3'] else frase_julgamento(r3)}, como em H1: a "
    "diferença existe, mas é desprezível na prática (d de Cohen da população de cerca de "
    f"{br(d_pop['spread'], 2)}) e o poder era de apenas {pct(r3['poder'], 1)}. O IC da diferença, "
    f"{faixa(r3['ic'], lambda v: br(v, 3))}, {'contém' if r3['contem'] else 'não contém'} o valor verdadeiro "
    f"({br(delta_pop['spread'], 3)})."
)
u4 = r4["limite_unilateral"]
st.markdown(
    f"**H4 (spread, unilateral).** {'Rejeitou' if r4['teste']['rejeita'] else 'Não rejeitou'} H0 "
    f"(p = {fp(r4['teste']['p_valor'])}). A diferença verdadeira tem o sinal que H4 esperava (N+NE pagam "
    f"menos), então H0 (δ ≥ 0) é falsa e a decisão é {frase_julgamento(r4)}, inofensivo pelo mesmo motivo de "
    f"H3, com poder de {pct(r4['poder'], 1)}: maior que o de H3 porque o teste unilateral junta todo o α na "
    "cauda que importa, mas ainda baixo. O IC bilateral "
    f"{faixa(r4['ic'], lambda v: br(v, 3))} {'contém' if r4['contem'] else 'não contém'} o δ verdadeiro; o "
    f"intervalo unilateral que combina com o teste, (−∞; {br(u4, 3)}], "
    f"{'também o contém' if delta_pop['spread'] <= u4 else 'não o contém'} ({br(delta_pop['spread'], 3)}) e "
    f"{'contém o 0, o que é coerente com não rejeitar H0' if u4 >= 0 else 'não contém o 0, coerente com rejeitar H0'}."
)
m5 = pop_grupo["prazo_amortizacao"]
st.markdown(
    f"**H5 (prazo, bilateral).** {'Rejeitou' if r5['teste']['rejeita'] else 'Não rejeitou'} H0 e a decisão "
    f"foi {frase_julgamento(r5)}: a diferença verdadeira de prazo é de {br(delta_pop['prazo_amortizacao'], 1)} "
    f"meses ({br(m5['N+NE']['media'], 1)} contra {br(m5['Demais']['media'], 1)}) e o poder era de "
    f"{pct(r5['poder'], 1)}. A estimativa ({br(r5['estimativa'], 1)} meses) ficou "
    f"{br(abs(r5['estimativa'] - delta_pop['prazo_amortizacao']), 1)} meses "
    f"{'acima' if r5['estimativa'] > delta_pop['prazo_amortizacao'] else 'abaixo'} da verdadeira, e o IC "
    f"{faixa(r5['ic'], lambda v: br(v, 1))} {'contém' if r5['contem'] else 'não contém'} o valor verdadeiro. "
    "Cuidado: parte da diferença pode vir de N+NE terem mais contratos de infraestrutura (limitação 4)."
)

# --- Variabilidade amostral: 1.000 sorteios, pré-calculados ----------------------------
sim = carregar_simulacao()
REP = sim["repeticoes"]
CHAVE_IC = {"H1": "pi", "H2": "pi", "H3": "spread", "H4": "spread", "H5": "prazo_amortizacao"}
NOME_IC = {"pi": "π (Wald)", "spread": "δ do spread (Welch)", "prazo_amortizacao": "δ do prazo (Welch)"}

st.subheader("Variabilidade amostral: e se o sorteio fosse outro?")
st.markdown(
    f"Tudo acima vem de **uma** amostra. Para ver o efeito da variabilidade amostral, repetimos o sorteio "
    f"**{inteiro(REP)} vezes**, com as sementes {inteiro(sim['semente_inicial'])} a "
    f"{inteiro(sim['semente_final'])} (nenhuma é a semente do trabalho), sempre com n = {inteiro(sim['n'])}, e "
    "refizemos os cinco testes em cada amostra, com as mesmas funções de cálculo. A conta foi feita uma vez, "
    "fora do app, e os resultados estão gravados no repositório (`dados/simulacao.json`, gerado por "
    "`ferramentas/simular_sorteios.py`). É um exercício de estudo: nenhum número oficial dos testes vem daqui."
)
tabela_estatica(pd.DataFrame([
    {
        "Teste": hid,
        "Poder pela fórmula": pct(R[hid]["poder"], 1),
        f"Amostras que rejeitam H0 (de {inteiro(REP)})": pct(sim["rejeicao"][hid], 1),
        "Intervalo de 95%": NOME_IC[CHAVE_IC[hid]],
        "Intervalos que contêm o valor verdadeiro": pct(sim["cobertura"][CHAVE_IC[hid]], 1),
    }
    for hid in HIP
]))
p_chapeus = [xi / sim["n"] for xi in sim["x_nne"]]
p_hat = R["H1"]["estimativa"]
p0_h1 = HIP["H1"]["p0_ou_delta0"]
grafico(g.distribuicao_p_chapeu(
    p_chapeus,
    {
        f"π verdadeiro da população ({pct(pi_pop, 2)})": pi_pop,
        f"p̂ da nossa amostra ({pct(p_hat, 2)})": p_hat,
        f"valor de H0 no teste H1 ({pct(p0_h1, 1)})": p0_h1,
    },
    f"A proporção amostral varia de sorteio para sorteio ({inteiro(REP)} sorteios)",
    f"Fonte: {inteiro(REP)} sorteios de n = {inteiro(sim['n'])} da população de {inteiro(len(pop))} contratos "
    "(BNDES, operações não automáticas, dados abertos)",
))
dp_p = float(pd.Series(p_chapeus).std(ddof=1))
ep_formula = math.sqrt(pi_pop * (1 - pi_pop) / sim["n"])
abaixo = sum(p < p_hat for p in p_chapeus) / REP
st.caption(
    f"Nos {inteiro(REP)} sorteios, p̂ foi de {pct(min(p_chapeus), 1)} a {pct(max(p_chapeus), 1)}, com desvio "
    f"padrão de {br(100 * dp_p, 2)} ponto percentual; a fórmula do erro padrão, com o π verdadeiro, dá "
    f"{br(100 * ep_formula, 2)}. O p̂ da nossa amostra ficou acima de {pct(abaixo, 0)} dos sorteios."
)

st.subheader("Fomos azarados?")
erros_ii = [hid for hid in R if R[hid]["julgamento"] == "erro tipo II"]
acertos = [hid for hid in R if R[hid]["julgamento"] == "acerto" and R[hid]["teste"]["rejeita"]]
numeros = {0: "Nenhum", 1: "Um", 2: "Dois", 3: "Três", 4: "Quatro", 5: "Os cinco"}


def lista(ids: list[str]) -> str:
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " e " + ids[-1]


if erros_ii:
    poderes = [R[hid]["poder"] for hid in erros_ii]
    partes = [f"{pct(sim['rejeicao'][hid], 1)} das amostras no {hid}" for hid in erros_ii]
    rejeicoes = partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " e " + partes[-1]
    st.markdown(
        f"{numeros[len(erros_ii)]} dos cinco testes ({lista(erros_ii)}) deram erro tipo II. Não foi azar: foi "
        "**falta de poder diante de diferenças verdadeiras muito pequenas**. Com poderes de "
        f"{pct(min(poderes), 1)} a {pct(max(poderes), 1)}, em {pct(1 - max(poderes), 0)} a "
        f"{pct(1 - min(poderes), 0)} das amostras possíveis o resultado seria o mesmo. A simulação confirma: "
        f"nos {inteiro(REP)} sorteios, H0 foi rejeitada em {rejeicoes}, perto do poder da fórmula."
        + (f" Os testes em que a diferença verdadeira é grande ({lista(acertos)}) rejeitaram, como deveriam."
           if acertos else "")
    )
st.markdown(
    "A lição: **não rejeitar H0 não quer dizer que a diferença é zero**. Quer dizer que a amostra não era "
    "grande o bastante para vê-la, e aqui as diferenças que escaparam são pequenas demais para importar na "
    "prática."
)
todos_contem = all(R[hid]["contem"] for hid in R)
coberturas = list(sim["cobertura"].values())
st.markdown(
    f"**E os intervalos?** Os três intervalos (o de π, que serve a H1 e H2; o de δ do spread, que serve a H3 "
    f"e H4; e o de δ do prazo, de H5) {'contêm' if todos_contem else 'nem todos contêm'} o valor verdadeiro. "
    f"Isso não é garantido: ao longo dos {inteiro(REP)} sorteios, os intervalos cobriram o valor verdadeiro "
    f"em {pct(min(coberturas), 1)} a {pct(max(coberturas), 1)} das amostras, um pouco acima de 95% porque os "
    "erros padrão ignoram a correção para população finita (limitação 8). Em "
    f"{pct(1 - max(coberturas), 1)} a {pct(1 - min(coberturas), 1)} das amostras o intervalo erraria, e sem a "
    "população à mão não saberíamos se o nosso é um desses casos. É isso que \"95% de confiança\" quer dizer: "
    "uma propriedade do método, ao longo de muitas amostras, e não uma probabilidade sobre o intervalo que "
    "calculamos."
)


# --- Limitações (desenho, seção 9) -----------------------------------------------------
st.header("Limitações")
st.markdown(
    "**O efeito da variabilidade amostral.** Cada amostra dá números um pouco diferentes, e é por isso que "
    "damos intervalos, não só um número. O poder baixo de H1, H3 e H4 também é variabilidade amostral: "
    "quando a diferença verdadeira é pequena perto do ruído da amostra, o teste quase nunca a distingue de zero."
)
st.markdown(
    r"""
1. O teste conta contratos; o BNDES e o IBGE falam em valor. Um contrato de R\$ 1 mil pesa como um de R\$ 10 bi. O Painel 1 mostra as duas leituras.
2. A base não tem as operações automáticas (via bancos repassadores), onde estão as MPME. Os números oficiais de participação regional incluem essas operações.
3. 2002 a 2025 junta governos, políticas de juros (TJLP, PSI, TLP) e Selic muito diferentes.
4. N+NE têm mais infraestrutura (60% contra 33%), que tem prazos mais longos por natureza. Uma diferença de prazo pode vir do setor, não da região.
5. Spreads 0 e amortizações 0 que parecem erro de registro foram mantidos.
6. As réguas são normativas: o BNDES não tem obrigação legal de seguir a régua populacional.
7. Descrevemos e testamos; não explicamos. Mostrar que N+NE têm contratos mais longos não diz por quê. O trabalho não afirma causa.
8. Fração amostrada de 14%. Os erros padrão ignoram a correção para população finita; os intervalos saem um pouco mais largos do que o necessário, do lado conservador.
"""
)
