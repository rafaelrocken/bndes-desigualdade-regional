"""Simula 1.000 sorteios da população e grava o resultado em `dados/simulacao.json`.

É a mesma conta da seção 8 do caderno ("e se o sorteio fosse outro?"): sorteia
1.000 amostras aleatórias simples de n = 913, com as sementes 1 a 1.000 (nenhuma é
a semente do trabalho, 20261112), e em cada uma refaz os cinco testes com as funções
do `calculo.py`. Grava, para cada teste, a fração de amostras que rejeitam H0 (a
comparar com o poder calculado pela fórmula) e, para cada um dos três intervalos
(π, δ do spread e δ do prazo), a fração que contém o valor verdadeiro da população.
Grava também o número de contratos N+NE em cada amostra, para o histograma de p̂.

A conta leva cerca de 40 segundos; por isso o app lê o arquivo e não simula ao abrir.
Nenhum número dos testes oficiais vem daqui: é um exercício sobre a variabilidade
amostral. Mesmas sementes, mesmo resultado.

Uso, a partir da pasta do app:
    python -X utf8 ferramentas/simular_sorteios.py
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

import calculo as c  # noqa: E402

SAIDA = RAIZ / "dados" / "simulacao.json"
REPETICOES = 1000
SEMENTE_INICIAL = 1
ALFA = 0.05


def simular(pop: pd.DataFrame, n: int) -> dict:
    """Refaz os cinco testes em `REPETICOES` amostras e devolve as frações e os x de cada sorteio."""
    hip = {h["id"]: h for h in c.HIPOTESES}
    pi_pop = float((pop["grupo"] == "N+NE").mean())
    delta = {
        var: float(pop.loc[pop["grupo"] == "N+NE", var].mean() - pop.loc[pop["grupo"] == "Demais", var].mean())
        for var in ("spread", "prazo_amortizacao")
    }

    rejeita = {hid: 0 for hid in hip}
    contem = {"pi": 0, "spread": 0, "prazo_amortizacao": 0}
    x_nne = []
    sementes = range(SEMENTE_INICIAL, SEMENTE_INICIAL + REPETICOES)
    for semente in sementes:
        s = c.sortear_amostra(pop, n=n, semente=semente)
        x = int((s["grupo"] == "N+NE").sum())
        x_nne.append(x)
        for hid in ("H1", "H2"):
            r = c.teste_z_proporcao(x, n, hip[hid]["p0_ou_delta0"], ALFA)
            rejeita[hid] += r["rejeita"]
        # o intervalo de Wald só depende da amostra: é o mesmo em H1 e H2
        contem["pi"] += r["ic_wald"][0] <= pi_pop <= r["ic_wald"][1]
        for hid in ("H3", "H4", "H5"):
            var = hip[hid]["variavel"]
            r = c.teste_welch(s.loc[s["grupo"] == "N+NE", var], s.loc[s["grupo"] == "Demais", var],
                              hip[hid]["alternativa"], ALFA)
            rejeita[hid] += r["rejeita"]
            if hid != "H4":  # H3 e H4 dividem o mesmo intervalo bilateral
                contem[var] += r["ic_diferenca"][0] <= delta[var] <= r["ic_diferenca"][1]

    return {
        "descricao": (
            "1.000 amostras aleatórias simples de n contratos, sementes 1 a 1.000, com os cinco testes "
            "refeitos em cada uma. rejeicao: fração das amostras que rejeitam H0 (alfa = 0,05). "
            "cobertura: fração dos intervalos de 95% que contêm o valor verdadeiro da população. "
            "x_nne: contratos N+NE em cada amostra (p-chapéu = x / n)."
        ),
        "gerado_por": "ferramentas/simular_sorteios.py",
        "repeticoes": REPETICOES,
        "semente_inicial": SEMENTE_INICIAL,
        "semente_final": SEMENTE_INICIAL + REPETICOES - 1,
        "n": n,
        "alfa": ALFA,
        "rejeicao": {hid: v / REPETICOES for hid, v in rejeita.items()},
        "cobertura": {chave: v / REPETICOES for chave, v in contem.items()},
        "x_nne": x_nne,
    }


def main() -> None:
    pop = pd.read_parquet(RAIZ / "dados" / "populacao.parquet")
    meta = json.loads((RAIZ / "dados" / "metadados.json").read_text(encoding="utf-8"))
    resultado = simular(pop, meta["n"])
    SAIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"gravado {SAIDA.name}: rejeição {resultado['rejeicao']}, cobertura {resultado['cobertura']}")


if __name__ == "__main__":
    main()
