"""
Gráficos do Home Guardian (Plotly) para a interpretação dos resultados.

Cada função fig_* recebe os dados já calculados pelas simulações e devolve uma
go.Figure. gerar_todos() executa tudo e salva em docs/figuras/ (8 figuras).

Uso (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.relatorios.graficos
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import RAIZ

NOMES = {"ingenua": "Ingênua (ordem de chegada)", "gulosa": "Gulosa", "pli": "PLI (ótimo)"}
CORES = {"ingenua": "#9aa0a6", "gulosa": "#f29900", "pli": "#1a73e8"}
TEMPLATE = "plotly_white"


def _layout(fig: go.Figure, titulo: str, x: str = None, y: str = None) -> go.Figure:
    fig.update_layout(title=titulo, template=TEMPLATE, legend_title_text="Estratégia",
                      font=dict(size=14), margin=dict(l=60, r=30, t=70, b=60))
    if x:
        fig.update_xaxes(title_text=x)
    if y:
        fig.update_yaxes(title_text=y)
    return fig


def fig_comparacao_cenarios(linhas: list[dict]) -> go.Figure:
    """Z por cenário e estratégia (saída de simulacao.cenarios.executar_todos)."""
    cenarios = list(dict.fromkeys(l["cenario"] for l in linhas))
    fig = go.Figure()
    for est in ("ingenua", "gulosa", "pli"):
        z = [next(l["z"] for l in linhas if l["cenario"] == c and l["estrategia"] == est) for c in cenarios]
        fig.add_bar(name=NOMES[est], x=cenarios, y=z, marker_color=CORES[est],
                    text=[f"{v:.1f}" for v in z], textposition="outside")
    fig.update_layout(barmode="group")
    return _layout(fig, "Função objetivo Z por cenário (menor é melhor)", "Cenário", "Z")


def fig_fronteira(curva: list[dict]) -> go.Figure:
    """Risco residual × custo conforme o orçamento (saída de curva_orcamento)."""
    fig = go.Figure()
    for est in ("ingenua", "gulosa", "pli"):
        pontos = sorted({(l[f"custo_{est}"], l[f"residual_{est}"]) for l in curva})
        modo = "lines+markers" if est == "pli" else "markers"
        fig.add_scatter(name=NOMES[est], x=[p[0] for p in pontos], y=[p[1] for p in pontos],
                        mode=modo, line=dict(shape="hv", width=3),
                        marker=dict(size=12 if est != "pli" else 9, color=CORES[est],
                                    symbol="diamond" if est == "ingenua" else "circle"),
                        hovertemplate="custo %{x}<br>risco residual %{y:.1f}<extra>" + NOMES[est] + "</extra>")
    return _layout(fig, "Fronteira de Pareto: risco residual × custo de intervenção (C3)",
                   "Custo total das ações", "Risco residual")


def fig_prazos(linhas: list[dict]) -> go.Figure:
    """Z conforme o fator aplicado aos prazos (saída de varrer_prazos)."""
    fig = go.Figure()
    for est in ("ingenua", "gulosa", "pli"):
        fig.add_scatter(name=NOMES[est], x=[l["fator_prazo"] for l in linhas],
                        y=[l[f"z_{est}"] for l in linhas], mode="lines+markers",
                        line=dict(color=CORES[est], width=3))
    return _layout(fig, "Sensibilidade aos prazos de resposta (C3)",
                   "Fator multiplicador dos prazos (1,0 = original)", "Z")


def fig_distribuicao_mc(res: dict[str, np.ndarray]) -> go.Figure:
    """Distribuição do ganho percentual de Z do PLI sobre cada heurística."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("PLI sobre a ingênua", "PLI sobre a gulosa"))
    for col, base in enumerate(("ingenua", "gulosa"), start=1):
        zb = res[f"z_{base}"]
        ganho = 100 * (zb - res["z_pli"]) / np.where(zb == 0, 1, zb)
        fig.add_histogram(x=ganho, nbinsx=30, marker_color=CORES[base], showlegend=False,
                          hovertemplate="ganho %{x:.0f}%<br>%{y} instâncias<extra></extra>",
                          row=1, col=col)
        fig.add_annotation(x=0.5 + (col - 1) * 0.5 - 0.02, y=1.0, xref="paper", yref="paper",
                           xanchor="right", yanchor="top", showarrow=False,
                           text=f"PLI melhor em {100 * (ganho > 1e-6).mean():.0f}%<br>ganho médio {ganho.mean():.1f}%")
    fig.update_xaxes(title_text="Ganho de Z do PLI (%)")
    fig.update_yaxes(title_text="Nº de instâncias", col=1)
    return _layout(fig, f"Monte Carlo: ganho do PLI em {len(res['z_pli'])} instâncias sorteadas")


def fig_mapa_calor(grade: dict, base: str = "gulosa", metrica: str = "ganho") -> go.Figure:
    """Mapa (K × B) do ganho médio (%) ou das vitórias (%) do PLI sobre `base`."""
    chave = f"{metrica}_{base}"
    z = grade[chave]
    rotulo = "Ganho médio de Z (%)" if metrica == "ganho" else "Instâncias em que o PLI vence (%)"
    fig = go.Figure(go.Heatmap(
        z=z, x=grade["orcamentos"], y=grade["capacidades"], colorscale="Blues",
        colorbar=dict(title=rotulo), text=np.round(z, 1), texttemplate="%{text}",
        hovertemplate="orçamento %{x}<br>atenção %{y}<br>%{z:.1f}<extra></extra>"))
    fig.update_yaxes(dtick=1)
    return _layout(fig, f"Onde o PLI se separa da {base}: {rotulo}",
                   "Orçamento B", "Capacidade de atenção K")


def fig_fronteira_regressao(modelo, X, y, limiares=(0.5,), fundo_probabilidade: bool = False) -> go.Figure:
    """
    Geometria analítica da regressão logística: leituras no plano (taxa_norm, acel_norm)
    e a reta P = limiar (a x + b y + c = 0). X = características, y = 1 perigo real /
    0 alarme falso. Com fundo_probabilidade=True, colore o plano pela probabilidade.
    """
    from ..modelo.geometria import fronteira_decisao

    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    xmin, xmax = float(X[:, 0].min()), float(X[:, 0].max())
    pad_x, pad_y = 0.05 * (xmax - xmin), 0.05 * np.ptp(X[:, 1])
    y0, y1 = float(X[:, 1].min() - pad_y), float(X[:, 1].max() + pad_y)

    fig = go.Figure()
    if fundo_probabilidade:
        gx, gy = np.linspace(xmin - pad_x, xmax + pad_x, 120), np.linspace(y0, y1, 120)
        GX, GY = np.meshgrid(gx, gy)
        pontos = np.column_stack([GX.ravel(), GY.ravel()])
        z = ((pontos - modelo.media) / modelo.desvio) @ modelo.w + modelo.b
        prob = (1.0 / (1.0 + np.exp(-z))).reshape(GX.shape)
        fig.add_trace(go.Heatmap(x=gx, y=gy, z=prob, zmin=0, zmax=1, colorscale="RdYlBu_r",
                                 opacity=0.45,
                                 colorbar=dict(title=dict(text="P(perigo)", side="top"), orientation="h",
                                               x=0.5, xanchor="center", y=-0.28, yanchor="top",
                                               len=0.6, thickness=14),
                                 hovertemplate="taxa %{x:.4f}<br>aceleração %{y:.4f}<br>P %{z:.2f}<extra></extra>"))

    for rotulo, nome, cor in ((0, "Alarme falso", "#9aa0a6"), (1, "Perigo real", "#d93025")):
        sel = y == rotulo
        fig.add_scatter(x=X[sel, 0], y=X[sel, 1], mode="markers", name=nome,
                        marker=dict(size=6, color=cor, opacity=0.55))

    for lim in limiares:
        a, b, c = fronteira_decisao(modelo, lim)
        if abs(b) < 1e-12:
            continue
        fig.add_scatter(x=[xmin, xmax], y=[-(a * xmin + c) / b, -(a * xmax + c) / b], mode="lines",
                        name=f"Fronteira P = {lim:.1f}".replace(".", ","),
                        line=dict(color="#1a73e8", width=3 if lim == 0.5 else 2,
                                  dash="solid" if lim == 0.5 else "dash"))
    fig.update_xaxes(range=[xmin - pad_x, xmax + pad_x])
    fig.update_yaxes(range=[y0, y1])
    _layout(fig, "Regressão logística: reta de decisão no plano taxa × aceleração",
            "Taxa de variação normalizada (1/min)", "Aceleração normalizada (1/min²)")
    fig.update_layout(legend_title_text="Classe")
    if fundo_probabilidade:                 
        fig.update_layout(margin=dict(l=60, r=30, t=70, b=140))
    return fig


def salvar(fig: go.Figure, pasta: Path, nome: str) -> list[str]:
    """Salva HTML e, se o kaleido estiver instalado, PNG. Devolve os arquivos criados."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    criados = []
    html = pasta / f"{nome}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    criados.append(str(html))
    try:
        png = pasta / f"{nome}.png"
        fig.write_image(png, width=1100, height=620, scale=2)
        criados.append(str(png))
    except Exception:
        pass         
    return criados


def gerar_todos(pasta=None, n_mc: int = 900, n_grade: int = 100) -> list[str]:
    from ..config import carregar_parametros
    from ..sensores.deteccao import gerar_base, treinar_modelo
    from ..simulacao.cenarios import executar_todos
    from ..simulacao.monte_carlo import executar as executar_mc
    from ..simulacao.sensibilidade import curva_orcamento, grade_orcamento_atencao, varrer_prazos

    pasta = Path(pasta) if pasta else RAIZ / "docs" / "figuras"
    params = carregar_parametros()
    modelo = treinar_modelo(params, n_por_tipo=500)          
    X, y = gerar_base(params, n_por_tipo=200, semente=7)      
    grade = grade_orcamento_atencao(n_instancias=n_grade)
    figuras = {
        "01_comparacao_cenarios": fig_comparacao_cenarios(executar_todos()),
        "02_fronteira_pareto": fig_fronteira(curva_orcamento()),
        "03_sensibilidade_prazos": fig_prazos(varrer_prazos()),
        "04_monte_carlo_ganho": fig_distribuicao_mc(executar_mc(n=n_mc)),
        "05_mapa_ganho_vs_gulosa": fig_mapa_calor(grade, "gulosa", "ganho"),
        "06_mapa_vitorias_vs_gulosa": fig_mapa_calor(grade, "gulosa", "vitorias"),
        "07_mapa_ganho_vs_ingenua": fig_mapa_calor(grade, "ingenua", "ganho"),
        "08_fronteira_decisao_regressao": fig_fronteira_regressao(modelo, X, y),
    }
    criados = []
    for nome, fig in figuras.items():
        criados += salvar(fig, pasta, nome)
    return criados


if __name__ == "__main__":
    for arquivo in gerar_todos():
        print(arquivo)