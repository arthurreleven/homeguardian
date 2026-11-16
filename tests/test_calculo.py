"""Cálculo simbólico (SymPy) confere com as versões numéricas."""
import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.modelo import calculo as c


def test_derivada_simbolica():
    assert sp.simplify(c.taxa_simbolica - c.r0 * sp.exp(c.k * c.t)) == 0


def test_taxa_exata_confere_com_diferenca_finita():
    h = 1e-5
    num = (c.valor(10 + h, 25, 3.0, 0.06) - c.valor(10 - h, 25, 3.0, 0.06)) / (2 * h)
    assert c.taxa_exata(10, 3.0, 0.06) == pytest.approx(num, rel=1e-5)


def test_tempo_critico_exato_atinge_o_valor_critico():
    tc = c.tempo_critico_exato(25, 3.0, 0.06, 250)
    assert c.valor(tc, 25, 3.0, 0.06) == pytest.approx(250, rel=1e-9)


def test_risco_acumulado_analitico_igual_ao_numerico():
    for tempo in (1.0, 5.0, 20.0):
        assert c.risco_acumulado(tempo, 9, 0.7, 0.06) == pytest.approx(
            c.risco_acumulado_numerico(tempo, 9, 0.7, 0.06), rel=1e-6)


def test_risco_acumulado_forma_fechada():
    assert c.risco_acumulado(10, 9, 0.7, 0.06) == pytest.approx(9 * 0.7 * (2.718281828 ** 0.6 - 1) / 0.06, rel=1e-6)