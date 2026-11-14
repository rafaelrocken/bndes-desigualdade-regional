"""Painel 1: análise descritiva da população (os 6.320 contratos)."""
import sys
from pathlib import Path

# A página também roda sozinha (testes, `streamlit run paginas/populacao.py`): põe a raiz do app no caminho.
_RAIZ = str(Path(__file__).resolve().parents[1])
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

from dados_app import carregar, carregar_ibge, carregar_malhas  # noqa: E402
from tela import (  # noqa: E402
    cartoes_descritivos,
    inteiro,
    painel_descritivo,
    seletor_nivel,
    voltar_para_base,
    cabecalho,
    rodape,
)

pop, amostra, meta = carregar()

voltar_para_base()
cabecalho(
    "Painel 1 · População",
    f"Onde estão os {inteiro(len(pop))} contratos do BNDES, e em que condições?",
    f"Todos os **{inteiro(len(pop))} contratos** da população. Os números aqui são os **valores verdadeiros**, "
    "que o Painel 3 usa para conferir a inferência feita com a amostra.",
)
cartoes_descritivos(pop, amostra, "populacao")
nivel = seletor_nivel("populacao")
painel_descritivo(pop, amostra, "populacao", carregar_ibge(), carregar_malhas(), nivel)

rodape()
