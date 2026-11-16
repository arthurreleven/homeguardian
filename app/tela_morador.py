"""
Tela do morador: letras grandes, poucos comandos, linguagem simples.

Mostra o estado geral da casa, um cartão por ambiente (com a pergunta e dois botões
quando há consulta), o tempo estimado, o contato de confiança e o que já aconteceu.
Os controles da simulação ficam num bloco à parte, porque não existiriam num produto real.
"""
from pathlib import Path

import streamlit as st

import sessao as S
from home_guardian.modelo.estados import Estado

# os estilos ficam em app/estilos.css
CSS = "<style>\n" + Path(__file__).with_name("estilos.css").read_text(encoding="utf-8") + "\n</style>"
COR = {"ok": "#2e7d32", "atencao": "#e69500", "critico": "#c62828", "info": "#1565c0"}

PERGUNTAS = {
    "fogao": ("O fogão está ligado há mais tempo que o habitual. Você está utilizando-o?",
              "✅ Sim, estou usando"),
    "vazamento": ("Há água fora do lugar na lavanderia. Está tudo sob controle?",
                  "✅ Sim, está tudo bem"),
    "sobreaquecimento": ("Um equipamento elétrico está mais quente que o normal. Você está utilizando-o?",
                         "✅ Sim, estou usando"),
}
ORDEM = {Estado.CONSULTA: 0, Estado.CRITICO: 1, Estado.ESCALADA: 2, Estado.EM_RESPOSTA: 3,
         Estado.ALERTA: 4, Estado.SUPRIMIDO: 5, Estado.RESOLVIDO: 6, Estado.NORMAL: 7}


def _card(titulo: str, texto: str, nivel: str) -> None:
    st.markdown(f'<div class="hg-card" style="--c:{COR[nivel]}"><h3>{titulo}</h3><p>{texto}</p></div>',
                unsafe_allow_html=True)


def _conteudo(m) -> tuple[str, str]:
    """(nível, texto) mostrado ao morador para o estado atual da ocorrência."""
    e = m.estado
    if e in (Estado.CONSULTA, Estado.ALERTA):
        return "atencao", PERGUNTAS[m.tipo][0]
    if e is Estado.SUPRIMIDO:
        return "info", "Entendido. Vamos continuar acompanhando."
    if e is Estado.ESCALADA:
        if m.acao_passiva:
            return "atencao", "Já enviamos um aviso para você. Por favor, verifique agora."
        return "atencao", "Estamos analisando qual é a melhor ação."
    if e is Estado.CRITICO:
        return "critico", f"Situação crítica. Estamos avisando {S.CONTATO}, seu contato de confiança."
    if e is Estado.EM_RESPOSTA:
        return "info", f"{S.ACOES.get(m.acao, 'Ação em andamento.')} Seguimos acompanhando."
    if e is Estado.RESOLVIDO:
        return "ok", "Tudo voltou ao normal."
    return "ok", "Nenhuma ocorrência identificada."


def _frase_log(h: dict) -> str:
    amb = S.AMBIENTES[h["tipo"]][0]
    p, d = h["para"], h["de"]
    if p == "ALERTA":
        return (f"{amb}: leitura acima do normal." if d == "NORMAL"
                else f"{amb}: ainda não temos certeza, vamos perguntar de novo.")
    return {
        "CONSULTA": f"{amb}: perguntamos se você está usando.",
        "SUPRIMIDO": f"{amb}: você confirmou, seguimos acompanhando.",
        "ESCALADA": f"{amb}: sem confirmação suficiente, avisamos você.",
        "CRITICO": f"{amb}: situação crítica, {S.CONTATO} foi avisada.",
        "EM_RESPOSTA": f"{amb}: {S.ACOES.get(h['acao'], 'ação em andamento.')}",
        "RESOLVIDO": f"{amb}: tudo voltou ao normal.",
        "NORMAL": f"{amb}: monitoramento normal.",
    }.get(p, f"{amb}: {p.lower()}.")


def _banner(g) -> None:
    ativas = S.ativas(g)
    criticas = [m for m in ativas if m.estado is Estado.CRITICO]
    tranquilos = len(S.AMBIENTES) - len(ativas)
    if criticas:
        nivel, titulo, sub = "critico", "🚨 Situação crítica", f"{S.CONTATO} está sendo avisada."
    elif ativas:
        n = len(ativas)
        nivel, titulo = "atencao", "⚠️ Atenção"
        sub = f"{n} {'ambiente precisa' if n == 1 else 'ambientes precisam'} da sua atenção."
    else:
        nivel, titulo = "ok", "✅ Casa monitorada"
        sub = "Tudo certo. Nenhuma ocorrência no momento."
    if ativas:
        sub += f" {tranquilos} sem ocorrência."
    st.markdown(f'<div class="hg-banner" style="--c:{COR[nivel]}"><h2>{titulo}</h2><p>{sub}</p></div>',
                unsafe_allow_html=True)


def render() -> None:
    s = S.sessao()
    g = s["g"]
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🏠 Home Guardian")
    _banner(g)

    # ocorrências, das mais urgentes para as menos
    for m in sorted(g.maquinas.values(), key=lambda m: ORDEM[m.estado]):
        ambiente, icone, _ = S.AMBIENTES[m.tipo]
        nivel, texto = _conteudo(m)
        _card(f"{icone} {ambiente}", texto, nivel)
        if m.estado in (Estado.CONSULTA, Estado.ESCALADA, Estado.CRITICO):
            ini = s["prazo_inicial"].get(m.id) or m.prazo_min or 1.0
            st.progress(max(0.0, min(1.0, m.prazo_min / ini)),
                        text=f"Tempo estimado até o perigo: cerca de {m.prazo_min:.0f} min")
        if m.estado is Estado.CONSULTA:
            c1, c2 = st.columns(2)
            c1.button(PERGUNTAS[m.tipo][1], key=f"sim_{m.id}_{m.n_consultas}",
                      on_click=S.responder, args=(m.id, "sim"))
            c2.button("🆘 Preciso de ajuda", key=f"ajuda_{m.id}_{m.n_consultas}",
                      type="primary", on_click=S.responder, args=(m.id, "ajuda"))

    # ambientes sem ocorrência
    for tipo, (ambiente, icone, _) in S.AMBIENTES.items():
        if tipo not in g.maquinas:
            _card(f"{icone} {ambiente}", "Nenhuma ocorrência identificada.", "ok")

    # contato de confiança
    avisada = any(m.estado is Estado.CRITICO or m.acao == "acionar_contato" for m in g.maquinas.values())
    _card("👤 Contato de confiança", f"{S.CONTATO}: " + ("avisada agora." if avisada else "disponível se precisar."),
          "critico" if avisada else "info")

    # o que já aconteceu
    if s["log"]:
        with st.expander("🕒 O que já aconteceu", expanded=True):
            for h in reversed(s["log"][-8:]):
                st.write(f"**{h['t']:.0f} min** · {_frase_log(h)}")

    st.caption("Protótipo acadêmico. Não substitui detectores de fumaça, de gás e demais dispositivos certificados.")

    # ------------------------------------------------------ controles da simulação
    with st.expander("🧪 Controles da simulação (não fazem parte da tela do morador)"):
        a, b, c = st.columns([3, 1, 1])
        preset = a.selectbox("Cenário", list(S.PRESETS))
        semente = b.number_input("Semente", min_value=0, value=7, step=1)
        c.write("")
        if c.button("▶ Iniciar"):
            S.iniciar(preset, int(semente))
            st.rerun()

        if g.maquinas:
            st.write(f"Relógio simulado: **{s['relogio']:.0f} min**")
            t1, t2, _ = st.columns([1, 1, 3])
            t1.button("⏱ +1 min", on_click=S.avancar, args=(1,))
            t2.button("⏱ +5 min", on_click=S.avancar, args=(5,))
            for m in g.maquinas.values():
                x1, x2, x3 = st.columns([2, 2, 2])
                x1.write(f"**{m.id}** · {m.estado.value}")
                if m.estado is Estado.CONSULTA:
                    x2.button("Morador não respondeu", key=f"sr_{m.id}_{m.n_consultas}",
                              on_click=S.responder, args=(m.id, "sem_resposta"))
                if m.estado not in (Estado.NORMAL, Estado.RESOLVIDO):
                    x3.button("Leitura normalizou", key=f"norm_{m.id}_{m.estado.value}",
                              on_click=S.normalizar, args=(m.id,))
        else:
            st.info("Escolha um cenário e clique em Iniciar.")