"""Máquina de estados: fluxos, transições inválidas, grafo e integração com o PLI."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_parametros
from home_guardian.modelo.estados import (ESTADOS, Estado, Evento, Gerenciador, Maquina,
                                          TransicaoInvalida, alcancaveis, diagrama_mermaid,
                                          fortemente_conexo, matriz_adjacencia)

P = carregar_parametros()


def nova(p0=0.5, prazo=12):
    m = Maquina("fogao", "fogao", 9, p0, prazo, P)
    m.leitura_acima_limiar()
    m.enviar_pergunta()
    return m


def test_sim_com_probabilidade_baixa_suprime():
    m = nova(0.5)                      
    assert m.responder("sim") is Estado.SUPRIMIDO


def test_sim_com_probabilidade_alta_reconsulta_e_depois_suprime():
    m = nova(0.7)                       
    assert m.responder("sim") is Estado.ALERTA
    m.enviar_pergunta()
    assert m.responder("sim") is Estado.SUPRIMIDO


def test_limite_de_consultas_escala():
    m = nova(0.9)                       
    m.responder("sim")
    m.enviar_pergunta()
    assert m.responder("sim") is Estado.ESCALADA


def test_ajuda_vai_para_critico_e_sem_resposta_para_escalada():
    assert nova().responder("ajuda") is Estado.CRITICO
    m = nova()
    assert m.timeout() is Estado.ESCALADA and m.probabilidade > 0.5


def test_transicao_invalida():
    m = Maquina("x", "fogao", 9, 0.5, 12, P)
    with pytest.raises(TransicaoInvalida):
        m.responder("sim")              
    with pytest.raises(TransicaoInvalida):
        m.aplicar(Evento.ACAO_SELECIONADA)


def test_leitura_normal_volta_ao_repouso_e_reseta():
    m = nova(0.7)
    m.responder("sim")
    m.leitura_normal()
    assert m.estado is Estado.NORMAL and m.n_consultas == 0 and m.probabilidade == 0.7


def test_grafo_fortemente_conexo_e_matriz():
    assert fortemente_conexo()
    assert alcancaveis(Estado.NORMAL) == set(ESTADOS)
    A = matriz_adjacencia()
    assert A.shape == (8, 8) and A[ESTADOS.index(Estado.NORMAL), ESTADOS.index(Estado.ALERTA)] == 1
    assert "stateDiagram-v2" in diagrama_mermaid()


def test_gerenciador_decide_com_pli_e_muda_estados():
    g = Gerenciador(P)
    for args in (("fogao", "fogao", 9, 0.7, 12), ("vazamento", "vazamento", 6, 0.8, 20),
                 ("sobreaquecimento", "sobreaquecimento", 8, 0.6, 9)):
        m = g.registrar(*args)
        m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
    r = g.decidir()
    assert r["custo"] <= P["orcamento"] and r["atencao"] == 0     
    for id_, acao in r["acoes"].items():
        assert g.maquinas[id_].estado is Estado.EM_RESPOSTA and g.maquinas[id_].acao == acao
    assert g.decidir() is None or all(m.estado is not Estado.NORMAL for m in g.a_decidir())


def test_gerenciador_sem_ocorrencias_a_decidir():
    g = Gerenciador(P)
    g.registrar("fogao", "fogao", 9, 0.7, 12)
    assert g.decidir() is None


def test_avancar_tempo_torna_acoes_lentas_inviaveis():
    g = Gerenciador(P)
    m = g.registrar("fogao", "fogao", 9, 0.7, 12)
    m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
    g.avancar_tempo(5)
    assert m.prazo_min == 7
    assert g.decidir()["acoes"]["fogao"] != "acionar_contato"

def test_ocorrencia_preterida_recebe_aviso_e_escala_por_prazo():
    g = Gerenciador(P)
    for args in (("fogao", "fogao", 9, 0.7, 12), ("vazamento", "vazamento", 6, 0.8, 20),
                 ("sobreaquecimento", "sobreaquecimento", 8, 0.6, 9)):
        m = g.registrar(*args)
        m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
    g.decidir()
    v = g.maquinas["vazamento"]
    assert v.estado is Estado.ESCALADA and v.acao is None
    assert v.acao_passiva == "notificar" and v.rodadas_sem_acao == 1
    assert v.como_ocorrencia()["gravidade"] > v.gravidade
    g.avancar_tempo(18)
    assert v.estado is Estado.CRITICO