"""
Tela técnica: mostra a matemática por trás das decisões.

Abas: ao vivo (KPIs, grafo de estados, decisões, sensores), cenários C1 a C6, sensibilidade,
Monte Carlo, mapa de calor (B × K), regressão logística, banco de dados e o modelo matemático.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sessao as S
from home_guardian.modelo.estados import ESTADOS, TRANSICOES, Estado
from home_guardian.persistencia.db import Banco
from home_guardian.relatorios import graficos as G
from home_guardian.simulacao.cenarios import CENARIOS, executar_todos
from home_guardian.simulacao.monte_carlo import executar as executar_mc
from home_guardian.simulacao.monte_carlo import formatar_resumo, resumir
from home_guardian.simulacao.sensibilidade import (curva_orcamento, grade_orcamento_atencao,
                                                   varrer_pesos, varrer_prazos)

COR_ESTADO = {"NORMAL": "#e0e0e0", "ALERTA": "#ffe082", "CONSULTA": "#ffe082", "SUPRIMIDO": "#b3e5fc",
              "ESCALADA": "#ffb74d", "CRITICO": "#ef9a9a", "EM_RESPOSTA": "#a5d6a7", "RESOLVIDO": "#c8e6c9"}


@st.cache_data(show_spinner="Rodando os cenários...")
def _cenarios():
    return executar_todos()


@st.cache_data(show_spinner="Calculando a curva de orçamento...")
def _curva(cenario: str):
    return curva_orcamento(cenario)


@st.cache_data
def _prazos(cenario: str):
    return varrer_prazos(nome_cenario=cenario)


@st.cache_data
def _pesos(parametro: str, cenario: str):
    valores = (0, 1, 3, 6, 10) if parametro == "lambda_custo" else (0, 1, 3, 6, 12)
    return varrer_pesos(parametro, valores, cenario)


def _acoes_txt(acoes: dict) -> str:
    return ", ".join(f"{i}: {j}" for i, j in acoes.items()) or "(nenhuma)"


def _dot_estados(g) -> str:
    """Grafo de estados em DOT; os estados em que há ocorrências ficam destacados."""
    ocupados = {}
    for m in g.maquinas.values():
        ocupados.setdefault(m.estado.value, []).append(m.id)
    linhas = ["digraph G {", 'bgcolor="white"; rankdir=LR; nodesep=0.35;',
              'node [shape=box, style="rounded,filled", fontname="Helvetica", fontcolor="black"];',
              'edge [fontsize=9, fontname="Helvetica", color="#555555", fontcolor="#333333"];']
    for e in ESTADOS:
        v = e.value
        rotulo = v + ("\\n" + ", ".join(ocupados[v]) if v in ocupados else "")
        pen = ', penwidth=3, color="#d32f2f"' if v in ocupados else ""
        linhas.append(f'{v} [label="{rotulo}", fillcolor="{COR_ESTADO[v]}"{pen}];')
    for (o, ev), d in TRANSICOES.items():
        linhas.append(f'{o.value} -> {d.value} [label="{ev.value}"];')
    linhas.append("}")
    return "\n".join(linhas)


def _aba_ao_vivo() -> None:
    s = S.sessao()
    g = s["g"]
    if not g.maquinas:
        st.info("Nenhuma simulação em andamento. Inicie uma na tela do morador (bloco de controles).")
        return
    st.write(f"Cenário: **{s['cenario']}** · relógio simulado: **{s['relogio']:.0f} min**")

    # KPIs
    ultima = s["decisoes"][-1]["resultado"] if s["decisoes"] else None
    k = st.columns(5)
    k[0].metric("Ocorrências ativas", len(S.ativas(g)))
    k[1].metric("Críticas", sum(m.estado is Estado.CRITICO for m in g.maquinas.values()))
    k[2].metric("Em resposta", sum(m.estado is Estado.EM_RESPOSTA for m in g.maquinas.values()))
    k[3].metric("Rodadas do PLI", s["rodada"])
    k[4].metric("Risco evitado (última)",
                f"{ultima['risco_inicial'] - ultima['risco_residual']:.1f}" if ultima else "n/d")

    st.subheader("Ocorrências")
    st.dataframe(pd.DataFrame([{
        "ocorrência": m.id, "estado": m.estado.value, "p inicial": round(m.probabilidade_inicial, 2),
        "p atual": round(m.probabilidade, 2), "prazo (min)": round(m.prazo_min, 1),
        "gravidade": m.gravidade, "gravidade efetiva": round(m.como_ocorrencia()["gravidade"], 2),
        "rodadas sem ação": m.rodadas_sem_acao, "ação": m.acao, "ação passiva": m.acao_passiva,
    } for m in g.maquinas.values()]), hide_index=True)

    st.subheader("Grafo de estados")
    st.caption("Estados com ocorrência aparecem com borda vermelha e o nome da ocorrência.")
    st.graphviz_chart(_dot_estados(g))

    st.subheader("Decisões do otimizador")
    if not s["decisoes"]:
        st.caption("Ainda não houve decisão: o PLI roda quando não há mais consultas pendentes ao morador.")
    for d in s["decisoes"]:
        r = d["resultado"]
        with st.expander(f"Rodada {d['rodada']} · t = {d['relogio']:.0f} min · Z = {r['z']:.1f}",
                         expanded=(d is s["decisoes"][-1])):
            c = st.columns(5)
            c[0].metric("Z", f"{r['z']:.1f}")
            c[1].metric("Risco inicial", f"{r['risco_inicial']:.1f}")
            c[2].metric("Risco residual", f"{r['risco_residual']:.1f}")
            c[3].metric("Custo / orçamento", f"{r['custo']:.0f} / {r['orcamento']:.0f}")
            c[4].metric("Atenção / capacidade", f"{r['atencao']:.0f} / {r['capacidade']:.0f}")
            st.write("Ações escolhidas: " + _acoes_txt(r["acoes"]))
            ids = list(r["risco_por_ocorrencia"])
            fig = go.Figure(go.Bar(x=ids, y=[r["risco_por_ocorrencia"][i] for i in ids],
                                   text=[r["acoes"].get(i, "sem ação") for i in ids], textposition="outside"))
            G._layout(fig, "Risco residual por ocorrência (rótulo = ação escolhida)", "Ocorrência", "Risco residual")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, key=f"res_{d['rodada']}")

    st.subheader("Linha do tempo")
    if s["log"]:
        st.dataframe(pd.DataFrame([{"t (min)": h["t"], "ocorrência": h["id"], "de": h["de"],
                                    "evento": h["evento"], "para": h["para"]} for h in s["log"]]),
                     hide_index=True)

    st.subheader("Leituras do sensor")
    if s["series"]:
        tipo = st.selectbox("Sensor", list(s["series"]))
        ser, perfil = s["series"][tipo], S.parametros()["sensores"]["perfis"][tipo]
        fig = go.Figure()
        fig.add_scatter(x=ser["t"], y=ser["x"], mode="lines", name="leitura (com ruído)")
        fig.add_scatter(x=ser["t"], y=ser["info"]["verdadeiro"], mode="lines", name="valor verdadeiro (simulado)",
                        line=dict(dash="dot"))
        fig.add_hline(y=perfil["limiar_alerta"], line_dash="dash", annotation_text="limiar de alerta")
        fig.add_hline(y=perfil["critico"], line_dash="dash", line_color="red", annotation_text="valor crítico")
        fig.add_vline(x=ser["alerta"], line_dash="dot", annotation_text="alerta detectado")
        tipo_serie = "perigosa" if ser["info"]["perigo"] else "benigna"
        G._layout(fig, f"{perfil['variavel']} (anomalia {tipo_serie})", "Tempo (min)", perfil["unidade"])
        st.plotly_chart(fig)

    with st.expander("Envelhecimento da gravidade (regra do vazamento esquecido)"):
        P = S.parametros()["estados"]
        alfa = P.get("envelhecimento", 0.25)
        ks = list(range(0, 9))
        fig = go.Figure()
        for gr in (4, 6, 8):
            fig.add_scatter(x=ks, y=[min(10, gr * (1 + alfa * k)) for k in ks], mode="lines+markers",
                            name=f"gravidade {gr}")
        G._layout(fig, f"g_eff = min(10, g·(1 + {alfa}·k))", "Rodadas sem ação (k)", "Gravidade efetiva")
        st.plotly_chart(fig, key="envelhecimento")


def _aba_cenarios() -> None:
    linhas = _cenarios()
    st.plotly_chart(G.fig_comparacao_cenarios(linhas))
    st.dataframe(pd.DataFrame([{
        "cenário": l["cenario"], "estratégia": l["estrategia"], "Z": round(l["z"], 1),
        "risco residual": round(l["risco_residual"], 1), "redução (%)": round(l["reducao_risco_pct"], 1),
        "custo": l["custo"], "atenção": l["atencao"], "ganho do PLI (%)": round(l["ganho_pli_pct"], 1),
        "tempo (ms)": round(l["tempo_ms"], 1), "ações": _acoes_txt(l["acoes"]),
    } for l in linhas]), hide_index=True)
    with st.expander("Descrição dos cenários"):
        for nome, desc in dict.fromkeys((l["cenario"], l["descricao"]) for l in linhas):
            st.write(f"**{nome}**: {desc}")


def _aba_sensibilidade() -> None:
    cen = st.selectbox("Cenário base", CENARIOS, index=CENARIOS.index("c3"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(G.fig_fronteira(_curva(cen)))
    c2.plotly_chart(G.fig_prazos(_prazos(cen)))
    st.subheader("Peso dos termos da função objetivo")
    par = st.radio("Parâmetro", ["lambda_custo", "lambda_falso_alarme"], horizontal=True)
    st.dataframe(pd.DataFrame([{
        par: l[par], "Z": round(l["z"], 1), "risco residual": round(l["risco_residual"], 1),
        "custo": l["custo"], "ações": _acoes_txt(l["acoes"])} for l in _pesos(par, cen)]), hide_index=True)


def _aba_monte_carlo() -> None:
    a, b, c = st.columns([2, 1, 1])
    n = a.slider("Instâncias sorteadas", 100, 2000, 300, 100)
    semente = b.number_input("Semente", min_value=0, value=42, step=1, key="mc_semente")
    c.write("")
    if c.button("Rodar Monte Carlo"):
        with st.spinner("Resolvendo o PLI em cada instância..."):
            st.session_state["mc"] = executar_mc(n=n, semente=int(semente))
    res = st.session_state.get("mc")
    if res is None:
        st.info("Clique em Rodar. Com 300 instâncias costuma levar alguns segundos.")
        return
    st.plotly_chart(G.fig_distribuicao_mc(res))
    st.code(formatar_resumo(resumir(res)), language=None)


def _aba_mapa() -> None:
    a, b = st.columns([2, 1])
    n = a.slider("Instâncias por célula", 20, 100, 30, 10)
    b.write("")
    if b.button("Calcular mapa"):
        with st.spinner("Isto resolve centenas de PLIs, pode demorar..."):
            st.session_state["grade"] = grade_orcamento_atencao(n_instancias=n)
    grade = st.session_state.get("grade")
    if grade is None:
        st.info("Clique em Calcular mapa.")
        return
    c1, c2 = st.columns(2)
    base = c1.radio("Comparar com", ["gulosa", "ingenua"], horizontal=True)
    metrica = c2.radio("Métrica", ["ganho", "vitorias"], horizontal=True,
                       format_func=lambda m: "Ganho médio de Z" if m == "ganho" else "% de vitórias")
    st.plotly_chart(G.fig_mapa_calor(grade, base, metrica))


def _aba_regressao() -> None:
    P, M = S.parametros(), S.modelo()
    st.write("Regressão logística: P(perigo real | taxa, aceleração), treinada com dados **sintéticos**.")
    c = st.columns(4)
    c[0].metric("AUC", f"{M.metricas['auc']:.2f}")
    c[1].metric("Acurácia", f"{M.metricas['acuracia']:.2f}")
    c[2].metric("Treino", M.metricas["n_treino"])
    c[3].metric("Teste", M.metricas["n_teste"])

    st.subheader("Ciclo com o banco")
    a, b = st.columns(2)
    semente = a.number_input("Semente do histórico", min_value=0, value=11, step=1)
    if a.button("Gerar 300 ocorrências sintéticas no SQLite"):
        with Banco() as banco:
            n = banco.gerar_historico(P, M, n=300, semente=int(semente))
        st.success(f"{n} ocorrências rotuladas gravadas.")
    if b.button("Retreinar com o histórico do SQLite"):
        with Banco() as banco:
            try:
                novo = banco.retreinar_do_banco()
                st.session_state["novo"] = novo.metricas
            except ValueError as e:
                st.warning(str(e))
    novo = st.session_state.get("novo")
    if novo:
        st.write(f"Retreinado: AUC **{novo['auc']:.2f}** · acurácia **{novo['acuracia']:.2f}** · "
                 f"{novo['n_treino']} treino / {novo['n_teste']} teste")

    with Banco() as banco:
        X, y = banco.dados_para_regressao()
    if len(y):
        fig = go.Figure()
        for rotulo, nome in ((1, "perigo real"), (0, "benigna")):
            fig.add_scatter(x=X[y == rotulo, 0], y=X[y == rotulo, 1], mode="markers", name=nome,
                            marker=dict(opacity=0.6))
        G._layout(fig, f"Dados do banco ({len(y)} ocorrências rotuladas)",
                  "taxa normalizada", "aceleração normalizada")
        st.plotly_chart(fig)
    st.caption("Todos os dados são sintéticos: as métricas medem a capacidade de recuperar o que foi "
               "simulado, não o desempenho em casas reais.")


def _aba_banco() -> None:
    with Banco() as banco:
        st.subheader("Resumo por tipo")
        st.caption("Inclui o histórico sintético gerado para a regressão.")
        st.dataframe(pd.DataFrame(banco.resumo_por_tipo()), hide_index=True)
        st.subheader("Z médio por estratégia (cenários gravados)")
        st.dataframe(pd.DataFrame(banco.media_por_estrategia()), hide_index=True)
        st.subheader("Últimas ocorrências")
        st.dataframe(pd.DataFrame(banco.ultimas_ocorrencias(30)), hide_index=True)


def _aba_modelo() -> None:
    arquivo = S.RAIZ / "docs" / "modelo_matematico.md"
    if arquivo.exists():
        st.markdown(arquivo.read_text(encoding="utf-8"))
    else:
        st.info("Salve o documento em docs/modelo_matematico.md para vê-lo aqui.")


def render() -> None:
    st.title("📊 Home Guardian · tela técnica")
    abas = st.tabs(["Ao vivo", "Cenários", "Sensibilidade", "Monte Carlo", "Mapa B × K",
                    "Regressão", "Banco de dados", "Modelo matemático"])
    for aba, f in zip(abas, (_aba_ao_vivo, _aba_cenarios, _aba_sensibilidade, _aba_monte_carlo,
                             _aba_mapa, _aba_regressao, _aba_banco, _aba_modelo)):
        with aba:
            f()