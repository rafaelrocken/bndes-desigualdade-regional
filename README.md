# O BNDES compensa a desigualdade regional?

Trabalho final de Probabilidade e Estatística para IA (IA003), da Especialização em Inteligência Artificial Avançada da UFRGS. É um aplicativo Streamlit que faz amostragem, intervalos de confiança e testes de hipóteses sobre as operações de financiamento não automáticas do BNDES, com os dados abertos do próprio banco.

> **Aviso.** Trabalho acadêmico independente, feito com dados abertos. Sem vínculo com o BNDES nem com o IBGE, e sem endosso deles.

Aplicativo: [URL_DO_APP](URL_DO_APP)

## A pergunta

O trabalho olha para a fatia dos contratos do BNDES que vai para o Norte e o Nordeste (N+NE) e a compara com três réguas:

| Régua | O que compara | Referência |
|---|---|---|
| PIB | a fatia dos contratos com a participação do N+NE no PIB | 19,6% (IBGE, Contas Regionais 2023: Nordeste 13,8% e Norte 5,8%) |
| População | a fatia dos contratos com a participação do N+NE na população | 35,4% (IBGE, Censo 2022: Nordeste 26,9% e Norte 8,5%) |
| Condições | o spread e o prazo de amortização dos contratos do N+NE com os dos demais estados | Política Nacional de Desenvolvimento Regional e Estratégia Nordeste 2023, que preveem condições diferenciadas para N e NE |

A população é o conjunto de 6.320 contratos (subcréditos reembolsáveis, com UF definida e sem as operações interestaduais, contratados de 2002 a 2025, agregados por contrato). A amostra é uma amostra aleatória simples de 913 contratos: margem de erro de 3 pontos percentuais, confiança de 95%, proporção suposta de 0,5 (o pior caso) e correção para população finita. Os parâmetros da população são calculados direto dela; a inferência usa só a amostra. O nível de significância é α = 0,05.

As cinco hipóteses, com π a proporção de contratos do N+NE na população e δ a diferença entre o spread médio do N+NE e o dos demais:

| | Teste | H0 | H1 | O que testa |
|---|---|---|---|---|
| H1 | Z de uma proporção | π = 0,196 | π ≠ 0,196 (bilateral) | a fatia dos contratos do N+NE é a do PIB? |
| H2 | Z de uma proporção | π = 0,354 | π ≠ 0,354 (bilateral) | a fatia dos contratos do N+NE é a da população? |
| H3 | t de Welch | δ = 0 | δ ≠ 0 (bilateral) | o spread médio do N+NE difere do dos demais? |
| H4 | t de Welch | δ ≥ 0 | δ < 0 (unilateral à esquerda) | o N+NE paga menos, como promete a política regional? |
| H5 | t de Welch | diferença dos prazos médios de amortização = 0 | diferença ≠ 0 (bilateral) | teste extra: o prazo médio do N+NE difere do dos demais? |

As definições completas estão em `calculo.py`, na lista `HIPOTESES`.

## Como navegar

O aplicativo abre na página "Base de dados e amostra", na raiz do endereço. Cada painel tem endereço próprio:

| Página | Endereço | O que traz |
|---|---|---|
| Base de dados e amostra | [URL_DO_APP](URL_DO_APP) | as três réguas, a base de dados, a população, o tamanho da amostra e o sorteio |
| Painel 1, população | [URL_DO_APP/populacao](URL_DO_APP/populacao) | onde estão os contratos, as condições (spread e prazo) e a qualidade dos dados, por região ou por estado |
| Painel 2, amostra | [URL_DO_APP/amostra](URL_DO_APP/amostra) | as mesmas figuras para a amostra, ao lado da população |
| Painel 3, inferência | [URL_DO_APP/inferencia](URL_DO_APP/inferencia) | intervalos de confiança, testes H1 a H5, comparação com a população, variabilidade em 1.000 sorteios e limitações |

Nos Painéis 1 e 2, o seletor "Ver por" alterna entre região e estado.

## Pré-registro

As hipóteses H1 a H5, α = 0,05 e a semente do sorteio (20261112) foram gravadas no commit `c47f903` de 12/11/2026 às 13h25, antes do sorteio, no repositório local do curso, que não é público. Não há link para ele. O hash completo, `c47f9033554cf2e07c9cac692754291c413c1d55`, está em `dados/metadados.json`.

A prova pública está no histórico deste repositório: as hipóteses (a lista `HIPOTESES` de `calculo.py`) entraram no commit [`fd7f39a`](https://github.com/rafaelrocken/bndes-desigualdade-regional/commit/fd7f39aa3fd677457f19eb9397165e2e3ef30a8b) em 12/11/2026, às 13h44, e o sorteio da amostra, no commit [`165e7c0`](https://github.com/rafaelrocken/bndes-desigualdade-regional/commit/165e7c0cf7323aacfe7cb570d2acd68af82e6ca2), às 13h49 do mesmo dia: as hipóteses vieram primeiro. Dá para conferir com `git log --format="%h %ad %s" --date=iso`.

## Fontes dos dados

- **BNDES**, operações de financiamento, operações não automáticas, no [Portal de Dados Abertos do BNDES](https://dadosabertos.bndes.gov.br/dataset/operacoes-financiamento). O arquivo usado é o [CSV das operações não automáticas](https://dadosabertos.bndes.gov.br/dataset/10e21ad1-568e-45e5-a8af-43f2c05ef1a2/resource/6f56b78c-510f-44b6-8274-78a5b7e931f4/download/operacoes-financiamento-operacoes-nao-automaticas.csv), atualizado no portal em 11/11/2026 e baixado em 12/11/2026 (24.035 linhas e 34 colunas).
- **IBGE**, PIB por estado em 2023: [SIDRA, tabela 5938](https://sidra.ibge.gov.br/tabela/5938) (Contas Regionais).
- **IBGE**, população por estado no Censo 2022: [SIDRA, tabela 4709](https://sidra.ibge.gov.br/tabela/4709).
- **IBGE**, malhas dos estados e das regiões para os mapas: [API de malhas](https://servicodados.ibge.gov.br/api/docs/malhas?versao=3).
- **Logos** (UFRGS, BNDES e IBGE): origem e condições de cada um em `assets/FONTES.md`.

## Como rodar

Precisa de Python 3.12 ou mais novo (é o que as versões fixadas de numpy e scipy exigem). Se for publicar no Streamlit Community Cloud, escolha o Python 3.13 nas configurações avançadas.

```bash
pip install -r requirements.txt
streamlit run app.py
```

O aplicativo lê só os arquivos de `dados/`, que já vêm no repositório. Não precisa de internet nem do CSV bruto.

### Testes

```bash
pip install pytest
python -m pytest -q
```

Os testes que leem o CSV bruto do BNDES o procuram em `../dados/operacoes-nao-automaticas.csv`, fora do repositório, e são pulados quando ele não está lá.

## Caderno de cálculo

O `caderno.ipynb` refaz, passo a passo, todas as contas do trabalho, com o texto ao lado. Ele vem sem saídas gravadas e usa as mesmas funções de `calculo.py` que o aplicativo. Os gráficos dele usam o matplotlib, que o aplicativo não usa e por isso não está no `requirements.txt`. Para rodá-lo:

```bash
pip install -r requirements.txt matplotlib jupyter
jupyter notebook caderno.ipynb
```

Abra o Jupyter a partir da pasta do repositório e rode as células de cima para baixo.

## Refazer os dados

Os arquivos de `dados/` já estão prontos; estes passos servem para quem quiser refazê-los. Todos os comandos partem da pasta do repositório.

| Para refazer | Comando | O que precisa |
|---|---|---|
| `dados/populacao.parquet`, `dados/amostra.parquet` e `dados/metadados.json` | `python -X utf8 preparar_dados.py <caminho do CSV>` | o CSV bruto do BNDES, que não vem no repositório: baixe pelo link da seção de fontes |
| `dados/ibge_uf.csv`, `dados/uf.geojson` e `dados/regiao.geojson` | `python -X utf8 ferramentas/baixar_ibge.py` | internet; confere que N+NE somam 19,6% do PIB e 35,4% da população |
| `dados/simulacao.json` | `python -X utf8 ferramentas/simular_sorteios.py` | refaz os 1.000 sorteios do Painel 3, em cerca de 40 segundos |
| `caderno.ipynb` | `python -X utf8 ferramentas/gerar_caderno.py` | `nbformat` |
| fotos das páginas | `python -X utf8 ferramentas/fotografar_paginas.py <url> <saida.png>` | o aplicativo rodando, um Chromium e o pacote `websockets`; o cabeçalho do script explica as opções |

Sobre o `preparar_dados.py`: o portal do BNDES atualiza o CSV de tempos em tempos, e o script confere que a base tem as 24.035 linhas e as 34 colunas da versão de 12/11/2026, parando com erro se for outra. Os dados de `dados/` são o retrato dessa data. Com o mesmo CSV e a semente 20261112, o sorteio sai igual.

## Estrutura de pastas

```
app.py               ponto de entrada: monta as quatro páginas
paginas/             uma página por arquivo (base, populacao, amostra, inferencia)
calculo.py           cálculo da população, da amostra e dos testes; as hipóteses
graficos.py          construtores dos gráficos
tela.py              cabeçalho, rodapé e blocos comuns às páginas
dados_app.py         leitura e cache dos dados
preparar_dados.py    monta a população e sorteia a amostra, uma única vez
caderno.ipynb        caderno de cálculo, sem saídas
dados/               população, amostra, metadados, simulação, PIB e população do IBGE, malhas
assets/              logos e assets/FONTES.md
ferramentas/         scripts para refazer os dados, o caderno e as fotos
tests/               testes automáticos
.streamlit/          tema do aplicativo
requirements.txt     dependências do aplicativo, com versões fixas
```

As tabelas de contratos em `dados/` trazem o número do contrato e o CNPJ do cliente, como na base aberta do BNDES.

## Licença

Os dados seguem os termos dos portais de origem: o Portal de Dados Abertos do BNDES e o IBGE. Os logos pertencem às respectivas instituições e estão registrados em `assets/FONTES.md`.
