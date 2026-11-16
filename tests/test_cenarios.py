"""Os cinco cenários rodam, respeitam seus overrides e o PLI nunca perde."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.simulacao.cenarios import CENARIOS, executar_cenario


@pytest.mark.parametrize("nome", CENARIOS)
def test_pli_melhor_ou_igual(nome):
    linhas = {l["estrategia"]: l for l in executar_cenario(nome)}
    assert linhas["pli"]["z"] <= linhas["ingenua"]["z"] + 1e-6
    assert linhas["pli"]["z"] <= linhas["gulosa"]["z"] + 1e-6


@pytest.mark.parametrize("nome", CENARIOS)
def test_recursos_respeitados(nome):
    for l in executar_cenario(nome):
        assert l["custo"] <= l["orcamento"]
        assert l["atencao"] <= l["capacidade"]


def test_c4_usa_orcamento_reduzido():
    assert all(l["orcamento"] == 5 for l in executar_cenario("c4"))


def test_c5_sem_atencao_do_morador():
    for l in executar_cenario("c5"):
        assert l["capacidade"] == 0 and l["atencao"] == 0