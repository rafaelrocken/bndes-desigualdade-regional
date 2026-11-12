"""Baixa do IBGE as réguas por estado e a malha do Brasil, e grava em `dados/`.

Gera três arquivos, todos de dados públicos do IBGE:

- `dados/ibge_uf.csv`: uma linha por UF com `uf, nome, regiao, pib_2023, populacao_2022`.
  O PIB está em mil reais, a preços correntes; a população é a residente no Censo 2022.
- `dados/uf.geojson`: a malha das 27 UFs, qualidade mínima, com as propriedades `uf`,
  `nome` e `regiao`. Os anéis ficam na orientação do d3-geo (contorno externo no sentido
  horário), a que o Vega usa para desenhar o mapa.
- `dados/regiao.geojson`: a malha das 5 grandes regiões, com `regiao` e `nome`.

Fontes (APIs públicas oficiais do IBGE):

- PIB por UF, 2023: SIDRA, tabela 5938 (Contas Regionais), variável 37
  https://apisidra.ibge.gov.br/values/t/5938/n3/all/v/37/p/2023
- População por UF, Censo 2022: SIDRA, tabela 4709, variável 93
  https://apisidra.ibge.gov.br/values/t/4709/n3/all/v/93/p/2022
- Nomes, siglas e regiões das UFs: API de localidades
  https://servicodados.ibge.gov.br/api/v1/localidades/estados
- Malhas: API de malhas v3, qualidade mínima
  https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo%2Bjson&qualidade=minima&intrarregiao=UF
  https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo%2Bjson&qualidade=minima&intrarregiao=regiao

Executar uma vez, de qualquer pasta:  python ferramentas/baixar_ibge.py
O script confere que Norte e Nordeste somam 19,6% do PIB e 35,4% da população,
as réguas de H1 e H2 (somando as partes de cada região arredondadas a 0,1%, como no
desenho do trabalho), e para com erro se não somarem.
"""
import csv
import gzip
import json
import urllib.request
from pathlib import Path

PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"

URL_PIB = "https://apisidra.ibge.gov.br/values/t/5938/n3/all/v/37/p/2023"
URL_POPULACAO = "https://apisidra.ibge.gov.br/values/t/4709/n3/all/v/93/p/2022"
URL_ESTADOS = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
URL_MALHA = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?formato=application/vnd.geo%2Bjson&qualidade=minima&intrarregiao={nivel}"
)
NOME_REGIAO = {"N": "Norte", "NE": "Nordeste", "CO": "Centro-Oeste", "SE": "Sudeste", "S": "Sul"}
# código IBGE da grande região (1 a 5) -> sigla
SIGLA_REGIAO = {"1": "N", "2": "NE", "3": "SE", "4": "S", "5": "CO"}


def _baixar_json(url: str):
    pedido = urllib.request.Request(url, headers={"User-Agent": "trabalho-academico-ia003"})
    with urllib.request.urlopen(pedido, timeout=120) as resposta:
        corpo = resposta.read()
    if corpo[:2] == bytes([0x1F, 0x8B]):  # a API de localidades responde comprimida em gzip
        corpo = gzip.decompress(corpo)
    return json.loads(corpo.decode("utf-8"))


def _sidra_por_uf(url: str) -> dict[str, int]:
    """Valor de cada UF (código IBGE de 2 dígitos) numa consulta da API do SIDRA."""
    linhas = _baixar_json(url)[1:]  # a primeira linha é o cabeçalho
    return {linha["D1C"]: int(linha["V"]) for linha in linhas}


def _arredondar(coordenadas):
    """Coordenadas com 3 casas (cerca de 100 m): basta para um mapa do país e encolhe o arquivo."""
    if isinstance(coordenadas[0], (int, float)):
        return [round(coordenadas[0], 3), round(coordenadas[1], 3)]
    return [_arredondar(c) for c in coordenadas]


def _area_com_sinal(anel) -> float:
    """Área (fórmula do laço) de um anel em graus; positiva no sentido anti-horário."""
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(anel, anel[1:])) / 2


def _orientar_para_o_d3(geometria: dict) -> None:
    """Põe os anéis na orientação que o d3-geo (usado pelo Vega) espera: o contorno
    externo no sentido horário e os buracos no anti-horário. A malha do IBGE vem na
    orientação contrária (a da RFC 7946), e com ela o Vega pinta o globo inteiro menos
    o estado, um retângulo cheio no lugar do mapa.
    """
    poligonos = [geometria["coordinates"]] if geometria["type"] == "Polygon" else geometria["coordinates"]
    for poligono in poligonos:
        for i, anel in enumerate(poligono):
            externo = i == 0
            if (_area_com_sinal(anel) > 0) == externo:  # externo anti-horário ou buraco horário
                anel.reverse()


def _malha(nivel: str, propriedades: dict[str, dict]) -> dict:
    """Malha do IBGE com as propriedades trocadas pelas nossas (por `codarea`)."""
    malha = _baixar_json(URL_MALHA.format(nivel=nivel))
    for feicao in malha["features"]:
        feicao["properties"] = propriedades[feicao["properties"]["codarea"]]
        feicao["geometry"]["coordinates"] = _arredondar(feicao["geometry"]["coordinates"])
        _orientar_para_o_d3(feicao["geometry"])
    return malha


def main() -> None:
    estados = _baixar_json(URL_ESTADOS)
    pib = _sidra_por_uf(URL_PIB)
    populacao = _sidra_por_uf(URL_POPULACAO)

    linhas = []
    for estado in sorted(estados, key=lambda e: e["sigla"]):
        codigo = str(estado["id"])
        linhas.append(
            {
                "uf": estado["sigla"],
                "nome": estado["nome"],
                "regiao": estado["regiao"]["sigla"],
                "pib_2023": pib[codigo],
                "populacao_2022": populacao[codigo],
                "_codigo": codigo,
            }
        )
    if len(linhas) != 27:
        raise SystemExit(f"esperava 27 UFs, vieram {len(linhas)}")

    # Conferência: as réguas de H1 e H2 saem destes números. O desenho do trabalho as
    # escreve como soma das partes de cada região arredondadas a 0,1% (PIB: NE 13,8% +
    # N 5,8% = 19,6%; população: NE 26,9% + N 8,5% = 35,4%), e é essa a conta conferida.
    # Sem arredondar as partes, a população dá 0,3546.
    def partes(coluna: str) -> dict[str, float]:
        total = sum(l[coluna] for l in linhas)
        return {r: sum(l[coluna] for l in linhas if l["regiao"] == r) / total for r in ("N", "NE")}

    for coluna, regua in (("pib_2023", 0.196), ("populacao_2022", 0.354)):
        p = partes(coluna)
        arredondada = round(round(p["N"], 3) + round(p["NE"], 3), 3)
        print(f"{coluna}: N {p['N']:.4f} + NE {p['NE']:.4f} = {p['N'] + p['NE']:.4f} "
              f"(partes a 0,1%: {arredondada:.3f}; régua {regua:.3f})")
        if arredondada != regua:
            raise SystemExit(f"{coluna}: a soma não bate com a régua {regua} de H1/H2")

    PASTA_DADOS.mkdir(exist_ok=True)
    with open(PASTA_DADOS / "ibge_uf.csv", "w", encoding="utf-8", newline="") as arquivo:
        arquivo.write(
            "# Fonte: IBGE, gerado por ferramentas/baixar_ibge.py\n"
            f"# pib_2023: PIB a preços correntes, mil reais, Contas Regionais 2023 (SIDRA 5938): {URL_PIB}\n"
            f"# populacao_2022: população residente, Censo 2022 (SIDRA 4709): {URL_POPULACAO}\n"
            f"# nome e regiao: API de localidades do IBGE: {URL_ESTADOS}\n"
        )
        escritor = csv.DictWriter(arquivo, fieldnames=["uf", "nome", "regiao", "pib_2023", "populacao_2022"],
                                  extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)

    por_codigo = {l["_codigo"]: {"uf": l["uf"], "nome": l["nome"], "regiao": l["regiao"]} for l in linhas}
    malha_uf = _malha("UF", por_codigo)
    regioes = {codigo: {"regiao": sigla, "nome": NOME_REGIAO[sigla]} for codigo, sigla in SIGLA_REGIAO.items()}
    malha_regiao = _malha("regiao", regioes)
    for nome, malha in (("uf.geojson", malha_uf), ("regiao.geojson", malha_regiao)):
        caminho = PASTA_DADOS / nome
        caminho.write_text(json.dumps(malha, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{caminho.name}: {len(malha['features'])} feições, {caminho.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
