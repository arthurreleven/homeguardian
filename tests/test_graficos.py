"""Os gráficos são construídos com os dados esperados e salvos em disco."""
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.relatorios.graficos import (fig_comparacao_cenarios, fig_fronteira,
                                               fig_mapa_calor, gerar_todos)
from home_guardian.simulacao.cenarios import CENARIOS, executar_todos
from home_guardian.simulacao.sensibilidade import curva_orcamento, grade_orcamento_atencao


def test_comparacao_tem_tres_estrategias_e_todos_os_cenarios():
    fig = fig_comparacao_cenarios(executar_todos())
    assert isinstance(fig, go.Figure) and len(fig.data) == 3
    assert all(len(t.x) == len(CENARIOS) for t in fig.data)


def test_fronteira_tem_tres_series():
    assert len(fig_fronteira(curva_orcamento("c3", range(0, 8))).data) == 3


def test_mapa_calor_dimensoes():
    g = grade_orcamento_atencao(n_instancias=10, orcamentos=(0, 6), capacidades=(0, 2, 4))
    z = fig_mapa_calor(g).data[0].z
    assert len(z) == 3 and len(z[0]) == 2


def test_gerar_todos_cria_arquivos(tmp_path):
    criados = gerar_todos(tmp_path, n_mc=20, n_grade=10)
    assert len([c for c in criados if c.endswith(".html")]) == 8
    assert all(Path(c).stat().st_size > 1000 for c in criados)