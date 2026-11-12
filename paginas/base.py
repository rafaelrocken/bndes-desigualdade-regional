"""Página "Base e amostra": a pergunta, a base, a população e o cálculo da amostra."""
import sys
from pathlib import Path

# A página também roda sozinha (testes, `streamlit run paginas/base.py`): põe a raiz do app no caminho.
_RAIZ = str(Path(__file__).resolve().parents[1])
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import calculo as c  # noqa: E402
from dados_app import carregar, carregar_ibge  # noqa: E402
from tela import (  # noqa: E402
    br,
    cabecalho,
    cartoes,
    inteiro,
    link_pagina,
    nota_regua_populacao,
    paginas_paineis,
    pct,
    rodape,
    tabela,
    tabela_estatica,
)

pop, amostra, meta = carregar()
ibge = carregar_ibge()
PAINEIS = paginas_paineis()
N = len(pop)
calc = c.tamanho_amostra(N, E=meta["E"], confianca=meta["confianca"], pi=meta["pi"])
n0, n, z = calc["n0"], calc["n"], calc["z"]


def em_latex(numero: str) -> str:
    """Vírgula decimal dentro de LaTeX: `{,}` evita o espaço que o LaTeX põe depois da vírgula."""
    return numero.replace(",", "{,}")


cabecalho()
st.title("Base de dados e amostra")
p0_pib, p0_pop = (h["p0_ou_delta0"] for h in c.HIPOTESES[:2])
with st.container(border=True):
    st.markdown(
        "O BNDES diz trabalhar pela redução das desigualdades regionais. Este trabalho mede essa promessa nos "
        f"**{inteiro(N)} contratos** do próprio banco, com três réguas:"
    )
    reguas = [
        (":material/account_balance:", "Régua do PIB",
         f"O Norte e o Nordeste recebem a sua parte na economia, **{pct(p0_pib)}** dos contratos?", "Teste H1"),
        (":material/groups:", "Régua da população",
         f"O Norte e o Nordeste recebem a sua parte nas pessoas, **{pct(p0_pop)}** dos contratos?", "Teste H2"),
        (":material/percent:", "Régua das condições",
         "O Norte e o Nordeste pagam um spread menor e têm prazo maior que as demais regiões?", "Testes H3, H4 e H5"),
    ]
    for coluna, (icone, nome, texto, teste) in zip(st.columns(3, gap="medium"), reguas):
        with coluna.container(border=True):
            st.markdown(f"{icone} **{nome}**  \n{texto}")
            st.caption(teste)
    st.markdown(
        "Este app responde em **quatro partes**: esta página apresenta a base de dados e o sorteio da amostra; "
        "o **Painel 1** descreve a população inteira; o **Painel 2** descreve a amostra, ao lado da população; e "
        "o **Painel 3** testa as cinco hipóteses só com a amostra e confere o resultado na população."
    )
    for coluna, (caminho, rotulo, icone) in zip(st.columns(3, gap="medium"), PAINEIS):
        with coluna.container(border=True):  # a borda faz do link um botão largo, fácil de achar
            link_pagina(caminho, rotulo, icone)
cartoes([
    {"rotulo": "Contratos na população (N)", "valor": inteiro(N),
     "ajuda": "Contratos reembolsáveis, com UF definida, de 2002 a 2025."},
    {"rotulo": "Contratos na amostra (n)", "valor": inteiro(n), "ajuda": "Amostra aleatória simples, sem reposição."},
    {"rotulo": "Linhas da base original", "valor": inteiro(meta["linhas_base"]),
     "ajuda": "Uma linha por subcrédito, antes de agregar em contratos."},
    {"rotulo": "Variáveis da base original", "valor": inteiro(meta["colunas_base"]), "ajuda": "Colunas do arquivo do BNDES."},
])
st.markdown(
    "O BNDES declara, na sua missão e nas Diretrizes Estratégicas 2026–2030, que trabalha pela "
    "\"redução das desigualdades sociais e regionais\". Este trabalho testa essa promessa com a base "
    "pública de operações não automáticas do banco, comparando o **Norte e Nordeste** (nas tabelas, N+NE) com as "
    "**demais regiões** (Centro-Oeste, Sudeste e Sul). Como a população inteira está disponível, "
    "os parâmetros verdadeiros são calculados nela, e a inferência usa só a amostra: no Painel 3 "
    "conferimos se os intervalos e os testes acertaram."
)

# As partes de cada região nas réguas, do IBGE, como o pré-registro as escreveu (a 0,1%).
partes = {
    medida: (ibge.groupby("regiao")[coluna].sum() / ibge[coluna].sum()).round(3)
    for medida, coluna in (("pib", "pib_2023"), ("populacao", "populacao_2022"))
}
x_nne = int((amostra["grupo"] == "N+NE").sum())

st.subheader("Três réguas de \"forma justa\"")
st.markdown(
    f"""
| Régua | Pergunta | Referência e fonte | Teste |
|---|---|---|---|
| **PIB** | Norte e Nordeste recebem a sua parte na economia? | {pct(p0_pib)} (IBGE, Contas Regionais 2023: NE {pct(partes["pib"]["NE"])} + N {pct(partes["pib"]["N"])}). O próprio BNDES usa esta régua: no Relatório Anual 2015 diz ter dado a N e NE mais que a sua parte no PIB | H1 |
| **População** | Norte e Nordeste recebem a sua parte nas pessoas? | {pct(p0_pop)} (IBGE, Censo 2022: NE {pct(partes["populacao"]["NE"])} + N {pct(partes["populacao"]["N"])}; soma das partes arredondadas, ver a nota abaixo). Régua normativa: a Constituição (art. 165, §7º) manda reduzir desigualdades inter-regionais "segundo critério populacional", e o ADCT (art. 35) fala em distribuição "proporcional à população". Vale para o orçamento federal, não para o crédito do BNDES; entra como parâmetro de justiça, não como obrigação do banco | H2 |
| **Condições** | Norte e Nordeste pagam menos e têm mais prazo? | Política Nacional de Desenvolvimento Regional (participação maior e garantias diferenciadas para N/NE) e Estratégia Nordeste 2023 (orçamento dedicado e condições diferenciadas) | H3, H4 e H5 |
"""
)
st.caption(nota_regua_populacao(ibge, x_nne, len(amostra)))
st.caption(
    "Base legal geral: Constituição Federal, art. 3º, III (objetivo de reduzir as desigualdades regionais) "
    "e art. 170, VII; o art. 165, §2º manda a lei de diretrizes orçamentárias fixar a política das "
    "agências financeiras oficiais de fomento."
)

st.subheader("A base de dados")
st.markdown(
    f"""
- **Fonte:** {meta["fonte_nome"]}.
- **Página do conjunto:** [{meta["fonte_url"]}]({meta["fonte_url"]})
- **Arquivo baixado:** [operacoes-financiamento-operacoes-nao-automaticas.csv]({meta["arquivo_url"]})
- **Atualizada no portal em** {pd.Timestamp(meta["atualizado_no_portal"]):%d/%m/%Y}; **baixada em** {pd.Timestamp(meta["baixado_em"]):%d/%m/%Y}.
- **Tamanho da base original:** {inteiro(meta["linhas_base"])} linhas (uma por subcrédito) × {inteiro(meta["colunas_base"])} colunas (variáveis).
"""
)
st.markdown(
    "**Contexto.** O BNDES (Banco Nacional de Desenvolvimento Econômico e Social) é o banco público "
    "federal de fomento. As operações **não automáticas** são as que o próprio BNDES analisa e aprova, "
    "contrato a contrato, em geral projetos maiores. As **automáticas** passam por bancos repassadores, "
    "que analisam o cliente em nome do BNDES, e não estão nesta base. A base existe para dar "
    "transparência às operações do banco; aqui ela serve para testar a promessa regional."
)
st.markdown(
    "**Subcrédito e contrato.** Em cada financiamento o banco assina um **contrato** com a empresa. "
    "O contrato se divide em **subcréditos**, parcelas definidas na assinatura, cada uma com valor, "
    "custo financeiro e prazo próprios. Subcrédito não é a liberação do dinheiro."
)

st.subheader("População e unidade amostral")
mais_de_um = int((pop["n_subcreditos"] > 1).sum())
st.markdown(
    f"""
- **Unidade amostral: o contrato.** A pergunta é sobre decisões de financiamento, e um contrato com 99 subcréditos não pode pesar 99 vezes. Por isso os subcréditos são agregados em contratos: uma linha por contrato ({inteiro(mais_de_um)} contratos, {pct(mais_de_um / N)}, têm mais de um subcrédito).
- **População:** contratos de modalidade reembolsável, com UF definida (saem as operações interestaduais, "IE") e contratados de 2002 a 2025. **N = {inteiro(N)} contratos.**
- **Grupos:** cada contrato fica na região da UF com a maior parte do valor; **Norte e Nordeste** (N+NE nas tabelas) de um lado; **demais regiões** (Centro-Oeste, Sudeste e Sul; Demais nas tabelas) do outro.
"""
)

st.markdown("**Variáveis usadas**")
tabela_estatica(
    pd.DataFrame(
        [
            ("grupo (N+NE ou Demais)", "qualitativa nominal", "evento de interesse de H1 e H2 (o contrato ser N+NE) e divisão dos grupos em H3 a H5"),
            ("regiao, uf", "qualitativa nominal", "onde está o contrato: a UF com a maior parte do valor e a sua região"),
            ("spread", "quantitativa contínua (% ao ano)", "juros acima do custo financeiro; H3 e H4"),
            ("prazo_amortizacao", "quantitativa contínua (meses)", "prazo para pagar o principal; H5"),
            ("valor_contratado", "quantitativa contínua (R$)", "tamanho do contrato; leitura por valor (limitação 1)"),
            ("setor", "qualitativa nominal", "setor do BNDES (infraestrutura, indústria, comércio e serviços, agropecuária); composição dos grupos"),
            ("porte", "qualitativa ordinal", "porte do cliente (micro a grande); composição dos grupos"),
        ],
        columns=["Variável", "Tipo", "Significado e papel no trabalho"],
    )
)

st.markdown("**Regras de agregação** (como vários subcréditos viram um contrato)")
tabela_estatica(
    pd.DataFrame(
        [
            ("região (N+NE ou Demais)", "a da UF com a maior parte do valor contratado (79 contratos têm mais de uma UF)"),
            ("valor contratado", "soma dos subcréditos"),
            ("spread (juros, % a.a.)", "média ponderada pelo valor dos subcréditos indexados em reais: TJLP e variantes (TJ462, TJ453, TJ6, TJ3), TLP, SELIC, IPCA, TR e CDI. Saem \"US$ / CESTA\" e \"TAXA FIXA EM US$\" (spread sobre moeda estrangeira não se compara com spread sobre TJLP), \"TAXA FIXA\" (ali o juro já é o custo total), \"SEM CUSTO\", \"% do CDI\" e \"OUTROS\". Contrato sem nenhum subcrédito indexado em reais fica fora de H3, H4 e da descritiva do spread"),
            ("prazo de amortização (meses)", "média ponderada pelo valor"),
            ("porte, setor, data, UF, CNPJ", "o do subcrédito de maior valor (131 contratos têm mais de um CNPJ e 301 mais de uma data; a regra resolve todos)"),
        ],
        columns=["Variável do contrato", "Regra"],
    )
)

st.subheader("Tamanho da amostra")
st.markdown("A fórmula para estimar uma proporção, com a correção para população finita:")
st.latex(r"n_0 = \frac{z^2\,\pi\,(1-\pi)}{E^2} \qquad\qquad n = \frac{n_0}{1 + n_0/N}")
st.markdown(
    f"""
- **Confiança de {pct(meta["confianca"], 0)}, logo $z = {em_latex(br(z, 2))}$.** É o padrão em pesquisa: o ponto da normal que deixa 2,5% de área em cada cauda.
- **Margem de erro $E = {br(100 * meta["E"], 0)}$ pontos percentuais.** Estreita o bastante para dizer se a proporção está perto ou longe das réguas de {pct(p0_pib)} e {pct(p0_pop)}.
- **$\\pi = {em_latex(br(meta["pi"], 1))}$, o caso conservador.** O produto $\\pi(1-\\pi)$ é máximo em 0,5, o que dá o maior $n$ possível; adequado porque não supomos nada sobre a proporção antes de ver os dados.
- **Correção para população finita**, porque vamos sortear uma fração grande ({pct(n / N)}) de uma população de {inteiro(N)} contratos.
"""
)
tabela(
    pd.DataFrame(
        [
            ("z (confiança de 95%)", br(z, 4)),
            ("z²", br(z**2, 4)),
            ("π (1 − π)", br(meta["pi"] * (1 - meta["pi"]), 4)),
            ("E²", br(meta["E"] ** 2, 4)),
            ("n₀ = z² · π (1 − π) / E²", br(n0, 1)),
            ("1 + n₀ / N", br(1 + n0 / N, 4)),
            ("n₀ / (1 + n₀ / N), antes de arredondar", br(n0 / (1 + n0 / N), 2)),
            ("n, arredondado para cima", inteiro(n)),
        ],
        columns=["Passo", "Valor"],
    )
)
st.markdown(f"**Tamanho da amostra: n = {inteiro(n)} contratos**, de N = {inteiro(N)}.")

st.subheader("Sorteio da amostra")
contagem = amostra["grupo"].value_counts().reindex(["N+NE", "Demais"], fill_value=0)
st.markdown(
    f"""
- **Amostragem aleatória simples sem reposição:** todo contrato tem a mesma chance de ser sorteado e nenhum entra duas vezes. A população é ordenada pelo número do contrato e sorteada com `sample(n={n}, replace=False, random_state={meta["semente"]})` do pandas.
- **Semente: {meta["semente"]}** (a data do sorteio). Com a mesma semente, qualquer pessoa repete o mesmo sorteio.
- **Pré-registro:** as cinco hipóteses, o nível de significância e a semente foram gravados no commit `{meta["commit_pre_registro"][:7]}` em 12/11/2026, às 13h25, no repositório local do curso, **antes** do sorteio. Assim o resultado do sorteio não pôde influenciar as hipóteses.
- **Resultado:** {inteiro(len(amostra))} contratos, {inteiro(contagem["N+NE"])} no Norte e Nordeste e {inteiro(contagem["Demais"])} nas demais regiões.
"""
)

st.subheader("Os painéis")
for caminho, rotulo, icone in PAINEIS:
    link_pagina(caminho, rotulo, icone)

rodape()
