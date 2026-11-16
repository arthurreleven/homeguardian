"""Propriedades esperadas da análise de sensibilidade."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.simulacao.sensibilidade import (curva_orcamento, grade_orcamento_atencao,
                                                   varrer_prazos, varrer_pesos)


def test_z_do_pli_nao_cresce_com_orcamento():
    z = [l["z_pli"] for l in curva_orcamento("c3", range(0, 17))]
    assert all(b <= a + 1e-6 for a, b in zip(z, z[1:]))


def test_custo_do_pli_respeita_orcamento():
    for l in curva_orcamento("c3", range(0, 17)):
        assert l["custo_pli"] <= l["orcamento"]


def test_lambda_custo_alto_reduz_custo():
    l = varrer_pesos("lambda_custo", (0, 10))
    assert l[1]["custo"] <= l[0]["custo"]


def test_prazos_mais_longos_nao_pioram_z():
    z = [l["z_pli"] for l in varrer_prazos((0.5, 1.0, 2.0))]
    assert z[2] <= z[1] + 1e-6 <= z[0] + 2e-6


def test_grade_pli_nunca_perde():
    g = grade_orcamento_atencao(n_instancias=30, orcamentos=(0, 6, 12), capacidades=(0, 2))
    assert np.all(g["ganho_gulosa"] >= -1e-6)
    assert np.all(g["ganho_ingenua"] >= -1e-6)