"""Fronteira de decisão da regressão logística como reta no plano."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_parametros
from home_guardian.modelo.geometria import distancia_a_fronteira, fronteira_decisao
from home_guardian.relatorios.fronteira_decisao import gerar
from home_guardian.sensores.deteccao import treinar_modelo

P = carregar_parametros()


@pytest.fixture(scope="module")
def modelo():
    return treinar_modelo(P, n_por_tipo=500, semente=11)


def test_pontos_da_reta_tem_probabilidade_meio(modelo):
    a, b, c = fronteira_decisao(modelo)
    for x1 in (0.01, 0.02, 0.05):
        x2 = -(a * x1 + c) / b
        assert modelo.probabilidade(x1, x2) == pytest.approx(0.5, abs=1e-9)
        assert distancia_a_fronteira(modelo, x1, x2) == pytest.approx(0.0, abs=1e-9)


def test_sinal_da_distancia_acompanha_o_lado(modelo):
    for tx, ac in ((0.001, 0.0), (0.08, 0.002), (0.03, -0.001)):
        assert (distancia_a_fronteira(modelo, tx, ac) > 0) == (modelo.probabilidade(tx, ac) > 0.5)


def test_outro_limiar_desloca_a_reta(modelo):
    assert fronteira_decisao(modelo, 0.8)[2] != fronteira_decisao(modelo)[2]


def test_figura_e_gerada(tmp_path):
    arquivo = Path(gerar(tmp_path))
    assert arquivo.exists() and arquivo.stat().st_size > 1000