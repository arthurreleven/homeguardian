"""
Figura da fronteira de decisão da regressão logística (Geometria Analítica).

Uso (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.relatorios.fronteira_decisao
Salva docs/figuras/08_fronteira_decisao_regressao.html
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ..config import RAIZ, carregar_parametros
from ..modelo.geometria import fronteira_decisao
from ..sensores.deteccao import gerar_base, treinar_modelo


def fig_fronteira_decisao(modelo, X: np.ndarray, y: np.ndarray, max_pontos: int = 600) -> go.Figure:
    idx = np.random.default_rng(0).permutation(len(y))[:max_pontos]
    fig = go.Figure()
    for rotulo, nome, cor in ((0, "Alarme falso", "#9aa0a6"), (1, "Perigo real", "#d93025")):
        sel = idx[y[idx] == rotulo]
        fig.add_scatter(x=X[sel, 0], y=X[sel, 1], mode="markers", name=nome,
                        marker=dict(color=cor, size=6, opacity=0.55))
    a, b, c = fronteira_decisao(modelo)
    xs = np.array([X[:, 0].min(), X[:, 0].max()])
    if abs(b) > 1e-12:
        fig.add_scatter(x=xs, y=-(a * xs + c) / b, mode="lines", name="Fronteira P = 0,5",
                        line=dict(color="#1a73e8", width=3))
        fig.update_yaxes(range=[float(X[:, 1].min()), float(X[:, 1].max())])
    fig.update_layout(title="Regressão logística: reta de decisão no plano taxa × aceleração",
                      template="plotly_white", font=dict(size=14), legend_title_text="Classe",
                      xaxis_title="Taxa de variação normalizada (1/min)",
                      yaxis_title="Aceleração normalizada (1/min²)")
    return fig


def gerar(pasta=None) -> str:
    params = carregar_parametros()
    X, y = gerar_base(params, n_por_tipo=300, semente=5)
    fig = fig_fronteira_decisao(treinar_modelo(params, n_por_tipo=500), X, y)
    pasta = Path(pasta) if pasta else RAIZ / "docs" / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "08_fronteira_decisao_regressao.html"
    fig.write_html(destino, include_plotlyjs="cdn")
    return str(destino)


if __name__ == "__main__":
    print(gerar())