"""Risco R = S * p * u, em escala de 0 a 100, com urgência u = min(1, d_ref / d)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_cenario, carregar_parametros
from home_guardian.modelo.otimizador import resolver
from home_guardian.modelo.risco import calcular_risco

P = carregar_parametros()


def test_valor_conhecido_do_fogao_no_c3():
    assert calcular_risco(9, 0.7, 12, 10) == pytest.approx(52.5)


def test_limites_da_escala():
    assert calcular_risco(10, 1.0, 5, 10) == pytest.approx(100.0)  
    assert calcular_risco(10, 0.0, 5, 10) == 0.0                   
    assert calcular_risco(0, 1.0, 5, 10) == 0.0                     


def test_urgencia_satura_em_um_para_prazos_curtos():
    r = calcular_risco(8, 0.6, 10, 10)
    assert calcular_risco(8, 0.6, 5, 10) == pytest.approx(r)
    assert calcular_risco(8, 0.6, 0.1, 10) == pytest.approx(r)


def test_prazo_maior_que_a_referencia_reduz_o_risco_proporcionalmente():
    base = calcular_risco(8, 0.6, 10, 10)
    assert calcular_risco(8, 0.6, 20, 10) == pytest.approx(base / 2)
    assert calcular_risco(8, 0.6, 40, 10) == pytest.approx(base / 4)


def test_monotonia():
    assert calcular_risco(9, 0.5, 12) > calcular_risco(5, 0.5, 12)      
    assert calcular_risco(5, 0.8, 12) > calcular_risco(5, 0.3, 12)      
    prazos = [2, 8, 10, 15, 30, 60]
    riscos = [calcular_risco(7, 0.5, d) for d in prazos]
    assert all(b <= a + 1e-12 for a, b in zip(riscos, riscos[1:]))       


def test_prazo_zero_nao_divide_por_zero():
    assert calcular_risco(8, 0.6, 0.0, 10) == pytest.approx(calcular_risco(8, 0.6, 10, 10))


def test_prazo_de_referencia_e_configuravel():
    assert calcular_risco(8, 0.6, 20, prazo_ref_min=20) == pytest.approx(calcular_risco(8, 0.6, 1, prazo_ref_min=20))
    assert calcular_risco(8, 0.6, 40, prazo_ref_min=20) == pytest.approx(calcular_risco(8, 0.6, 20, 20) / 2)


def test_risco_inicial_do_otimizador_no_c3():
    oc = carregar_cenario("c3")["ocorrencias"]
    esperado = sum(calcular_risco(o["gravidade"], o["probabilidade"], o["prazo_min"],
                                  P["prazo_referencia_min"]) for o in oc)
    assert esperado == pytest.approx(124.5)                              
    assert resolver(oc, P)["risco_inicial"] == pytest.approx(esperado)