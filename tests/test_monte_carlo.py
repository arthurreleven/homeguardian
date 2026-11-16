"""Monte Carlo: reprodutível, viável e com PLI nunca pior que as heurísticas."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.simulacao.monte_carlo import executar, resumir


def test_reprodutivel_com_mesma_semente():
    a, b = executar(n=30, semente=1), executar(n=30, semente=1)
    assert np.array_equal(a["z_pli"], b["z_pli"])


def test_pli_nunca_pior_e_recursos_respeitados():
    r = executar(n=150, semente=3)
    assert np.all(r["z_pli"] <= r["z_ingenua"] + 1e-6)
    assert np.all(r["z_pli"] <= r["z_gulosa"] + 1e-6)
    for nome in ("ingenua", "gulosa", "pli"):
        assert np.all(r[f"custo_{nome}"] <= r["orcamento"])
    s = resumir(r)
    assert s["vs_ingenua"]["pli_pior_pct"] == 0 and s["vs_gulosa"]["pli_pior_pct"] == 0