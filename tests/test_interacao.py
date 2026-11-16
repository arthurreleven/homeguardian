"""Atualização bayesiana e integração com os cenários."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_parametros
from home_guardian.modelo.interacao import (aplicar_respostas, atualizar_probabilidade,
                                            capacidade_atencao_efetiva, deve_suprimir)

P = carregar_parametros()


def test_valor_conhecido_de_bayes():
    assert atualizar_probabilidade(0.7, "sim", P) == pytest.approx(0.14 / 0.38)


def test_direcao_do_efeito():
    p0 = 0.5
    assert atualizar_probabilidade(p0, "sim", P) < p0
    assert atualizar_probabilidade(p0, "ajuda", P) > p0
    assert atualizar_probabilidade(p0, "sem_resposta", P) > p0


def test_extremos_e_validacao():
    assert atualizar_probabilidade(0.0, "ajuda", P) == 0.0
    assert atualizar_probabilidade(1.0, "sim", P) == 1.0
    with pytest.raises(ValueError):
        atualizar_probabilidade(0.5, "talvez", P)
    with pytest.raises(ValueError):
        atualizar_probabilidade(1.5, "sim", P)


def test_duas_respostas_sim_reduzem_mais():
    p1 = atualizar_probabilidade(0.7, "sim", P)
    assert atualizar_probabilidade(p1, "sim", P) < p1


def test_aplicar_respostas_nao_altera_original_e_guarda_prior():
    oc = [{"id": "a", "probabilidade": 0.7}, {"id": "b", "probabilidade": 0.4}]
    novo = aplicar_respostas(oc, {"a": "sim"}, P)
    assert oc[0]["probabilidade"] == 0.7
    assert novo[0]["probabilidade_inicial"] == 0.7 and novo[0]["probabilidade"] < 0.7
    assert novo[1]["probabilidade"] == 0.4


def test_capacidade_efetiva():
    assert capacidade_atencao_efetiva({"a": "sem_resposta", "b": "sem_resposta"}, P) == 0
    assert capacidade_atencao_efetiva({"a": "sem_resposta", "b": "sim"}, P) == P["capacidade_atencao"]
    assert capacidade_atencao_efetiva({}, P) == P["capacidade_atencao"]


def test_supressao():
    assert deve_suprimir(0.2, P) and not deve_suprimir(0.5, P)


def test_resposta_sim_evita_cortar_o_gas_no_c6():
    from home_guardian.simulacao.cenarios import executar_cenario
    pli = {l["estrategia"]: l for l in executar_cenario("c3")}["pli"]
    pli6 = {l["estrategia"]: l for l in executar_cenario("c6")}["pli"]
    assert pli["acoes"]["fogao"] == "cortar_gas"         
    assert pli6["acoes"]["fogao"] != "cortar_gas"         


def test_c5_sem_resposta_corta_gas_sem_depender_do_morador():
    from home_guardian.simulacao.cenarios import executar_cenario
    pli5 = {l["estrategia"]: l for l in executar_cenario("c5")}["pli"]
    assert pli5["acoes"]["fogao"] == "cortar_gas" and pli5["atencao"] == 0