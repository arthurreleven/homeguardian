"""
Estado e ações do app (Streamlit). Fica entre as telas e o motor de decisão.

As telas só leem o estado e chamam as funções daqui; o motor (Gerenciador, Maquina,
otimizador) não sabe que existe interface.

Regra de decisão: o PLI roda quando não há mais nenhuma consulta pendente ao morador,
assim todas as ocorrências escaladas são decididas juntas, dividindo o mesmo orçamento.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import numpy as np
import streamlit as st

from home_guardian.config import carregar_parametros
from home_guardian.modelo.estados import Estado, Gerenciador
from home_guardian.persistencia.db import Banco
from home_guardian.sensores.deteccao import analisar_serie, treinar_modelo
from home_guardian.sensores.simulador import gerar_serie

# tipo -> (ambiente, ícone, nome do equipamento)
AMBIENTES = {
    "fogao": ("Cozinha", "🍳", "fogão"),
    "vazamento": ("Lavanderia", "💧", "vazamento de água"),
    "sobreaquecimento": ("Área elétrica", "⚡", "equipamento elétrico"),
}

ACOES = {
    "notificar_morador": "Enviamos um aviso para você.",
    "alarme_sonoro": "Acionamos o alarme sonoro.",
    "acionar_contato": "Avisamos o seu contato de confiança.",
    "cortar_gas": "Fechamos o gás do fogão.",
    "fechar_valvula_agua": "Fechamos o registro de água.",
    "cortar_energia_equip": "Desligamos a energia do equipamento.",
}

CONTATO = "Maria (filha)"

PRESETS = {
    "Três perigos ao mesmo tempo": {"fogao": True, "vazamento": True, "sobreaquecimento": True},
    "Fogão e vazamento": {"fogao": True, "vazamento": True},
    "Panela em uso normal (alarme falso)": {"fogao": False},
    "Aleatório": None,
}


@st.cache_resource
def parametros() -> dict:
    return carregar_parametros()


@st.cache_resource(show_spinner="Treinando a regressão logística...")
def modelo():
    return treinar_modelo(parametros(), n_por_tipo=500)


def _novo(cenario: str = "") -> dict:
    return {"g": Gerenciador(parametros()), "relogio": 0.0, "extras": {}, "series": {},
            "decisoes": [], "rodada": 0, "sessao_id": None, "cenario": cenario,
            "prazo_inicial": {}, "log": [], "vistos": {}}


def sessao() -> dict:
    if "hg" not in st.session_state:
        st.session_state.hg = _novo()
    return st.session_state.hg


def pendentes(g: Gerenciador) -> list:
    """Ocorrências esperando pergunta ou resposta do morador."""
    return [m for m in g.maquinas.values() if m.estado in (Estado.CONSULTA, Estado.ALERTA)]


def ativas(g: Gerenciador) -> list:
    return [m for m in g.maquinas.values()
            if m.estado not in (Estado.NORMAL, Estado.SUPRIMIDO, Estado.RESOLVIDO)]


# ---------------------------------------------------------------- log e persistência
def _sincronizar_log() -> None:
    """Copia as transições novas de cada máquina para o log de eventos do app."""
    s = sessao()
    for m in s["g"].maquinas.values():
        vistos = s["vistos"].get(m.id, 0)
        for h in m.historico[vistos:]:
            s["log"].append({"t": s["relogio"], "id": m.id, "tipo": m.tipo, "de": h["de"].value,
                             "para": h["para"].value, "evento": h["evento"].value, "acao": m.acao})
        s["vistos"][m.id] = len(m.historico)


def salvar(resultado: dict | None = None) -> None:
    s = sessao()
    _sincronizar_log()
    with Banco() as b:
        if s["sessao_id"] is None:
            s["sessao_id"] = b.criar_sessao(f"app: {s['cenario'] or 'sem cenário'}")
        b.salvar_gerenciador(s["sessao_id"], s["g"], resultado, s["extras"])


def _decidir_se_pronto() -> dict | None:
    s = sessao()
    g = s["g"]
    if pendentes(g) or not g.a_decidir():
        return None
    r = g.decidir()
    s["rodada"] += 1
    s["decisoes"].append({"rodada": s["rodada"], "relogio": s["relogio"], "resultado": r})
    return r


# ----------------------------------------------------------------------- ações
def iniciar(preset: str, semente: int) -> None:
    """Começa uma nova simulação: gera as leituras, detecta e abre a consulta ao morador."""
    P, M = parametros(), modelo()
    rng = np.random.default_rng(int(semente))
    s = _novo(preset)
    escolha = PRESETS[preset]
    for tipo in P["sensores"]["perfis"]:
        if escolha is None:
            perigo = bool(rng.random() < P["sensores"]["prevalencia_perigo"])
        elif tipo in escolha:
            perigo = escolha[tipo]
        else:
            continue
        t, x, info = gerar_serie(tipo, P, rng, perigo)
        a = analisar_serie(tipo, t, x, P, M)
        if a is None:                       # nada foi detectado nessa série
            continue
        oc, c = a["ocorrencia"], a["caracteristicas"]
        m = s["g"].registrar(oc["id"], oc["tipo"], oc["gravidade"], oc["probabilidade"], oc["prazo_min"])
        m.leitura_acima_limiar()
        m.enviar_pergunta()
        s["prazo_inicial"][tipo] = oc["prazo_min"]
        s["series"][tipo] = {"t": t, "x": x, "info": info, "alerta": a["instante_alerta_min"]}
        s["extras"][tipo] = {"taxa": c["taxa"], "aceleracao": c["aceleracao"],
                             "taxa_norm": c["taxa_norm"], "acel_norm": c["acel_norm"],
                             "perigo_real": info["perigo"]}
    st.session_state.hg = s
    salvar()


def responder(ident: str, resposta: str) -> None:
    """Resposta do morador ("sim", "ajuda") ou "sem_resposta"."""
    m = sessao()["g"].maquinas[ident]
    m.responder(resposta)
    if m.estado is Estado.ALERTA:           # "sim", mas a probabilidade ainda está alta: reconsulta
        m.enviar_pergunta()
    salvar(_decidir_se_pronto())


def avancar(minutos: float) -> None:
    s = sessao()
    s["g"].avancar_tempo(minutos)
    s["relogio"] += minutos
    salvar(_decidir_se_pronto())


def normalizar(ident: str) -> None:
    """A leitura do sensor voltou ao normal."""
    sessao()["g"].maquinas[ident].leitura_normal()
    salvar(_decidir_se_pronto())