"""A formulação matricial (NumPy) confere com o PuLP."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_cenario, carregar_parametros
from home_guardian.modelo.interacao import aplicar_respostas
from home_guardian.modelo.matrizes import (construir_matrizes, matriz_de_decisao, valor_objetivo, viavel)
from home_guardian.modelo.otimizador import resolver

P = carregar_parametros()


def ocorrencias(nome):
    c = carregar_cenario(nome)
    return aplicar_respostas(c["ocorrencias"], c.get("respostas", {}), P)


@pytest.mark.parametrize("nome", ["c1", "c2", "c3", "c4", "c5", "c6"])
@pytest.mark.parametrize("B", [0, 5, 10, 13])
def test_z_matricial_igual_ao_do_pli(nome, B):
    oc = ocorrencias(nome)
    r = resolver(oc, P, orcamento=B)
    M = construir_matrizes(oc, P)
    X = matriz_de_decisao(M, r["acoes"])
    assert valor_objetivo(M, X, P) == pytest.approx(r["z"], abs=1e-9)
    assert viavel(M, X, P, orcamento=B)


def test_dimensoes_e_mascara_de_viabilidade():
    M = construir_matrizes(ocorrencias("c3"), P)
    assert M["E"].shape == M["C"].shape == M["F"].shape == (3, 6)
    i, j = M["ids"].index("sobreaquecimento"), M["acoes"].index("acionar_contato")
    assert M["F"][i, j] == 0                      
    assert np.all((M["E"] >= 0) & (M["E"] <= 1))


def test_viavel_detecta_violacoes():
    M = construir_matrizes(ocorrencias("c3"), P)
    gas = matriz_de_decisao(M, {"fogao": "cortar_gas", "vazamento": "fechar_valvula_agua",
                                "sobreaquecimento": "cortar_energia_equip"})
    assert not viavel(M, gas, P, orcamento=10)                    
    assert viavel(M, gas, P, orcamento=13)
    duas = matriz_de_decisao(M, {"fogao": "acionar_contato", "vazamento": "acionar_contato"})
    assert not viavel(M, duas, P, orcamento=20)                   
    inviavel = matriz_de_decisao(M, {"sobreaquecimento": "acionar_contato"})
    assert not viavel(M, inviavel, P, orcamento=20)               
    dupla = matriz_de_decisao(M, {"fogao": "cortar_gas"})
    dupla[0, 0] = 1                                               
    assert not viavel(M, dupla, P, orcamento=20)