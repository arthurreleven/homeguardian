"""As heurísticas devem ser viáveis e nunca melhores que o ótimo do PLI."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_cenario, carregar_parametros
from home_guardian.modelo.heuristica import gulosa, ingenua
from home_guardian.modelo.otimizador import resolver

ORCAMENTOS = [0, 2, 5, 8, 9, 10, 13, 20]


@pytest.mark.parametrize("B", ORCAMENTOS)
@pytest.mark.parametrize("estrategia", [ingenua, gulosa])
def test_heuristica_viavel(estrategia, B):
    p, c = carregar_parametros(), carregar_cenario("c3")
    r = estrategia(c["ocorrencias"], p, orcamento=B)
    assert r["custo"] <= B
    assert r["atencao"] <= p["capacidade_atencao"]


@pytest.mark.parametrize("B", ORCAMENTOS)
@pytest.mark.parametrize("estrategia", [ingenua, gulosa])
def test_pli_nunca_pior_que_heuristica(estrategia, B):
    p, c = carregar_parametros(), carregar_cenario("c3")
    z_pli = resolver(c["ocorrencias"], p, orcamento=B)["z"]
    z_h = estrategia(c["ocorrencias"], p, orcamento=B)["z"]
    assert z_pli <= z_h + 1e-6