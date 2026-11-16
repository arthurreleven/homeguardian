"""Simulador, detecção, derivada, regressão logística e integração com o motor."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_parametros
from home_guardian.modelo.estados import Estado, Gerenciador
from home_guardian.sensores.deteccao import (analisar_serie, caracteristicas, indice_de_alerta,
                                             prazo_ate_critico, treinar_modelo)
from home_guardian.sensores.simulador import gerar_serie

P = carregar_parametros()
S = P["sensores"]
TIPOS = list(S["perfis"])


@pytest.fixture(scope="module")
def modelo():
    return treinar_modelo(P, n_por_tipo=500, semente=11)


def test_serie_reprodutivel():
    a = gerar_serie("fogao", P, np.random.default_rng(1), True)[1]
    b = gerar_serie("fogao", P, np.random.default_rng(1), True)[1]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("tipo", TIPOS)
def test_benigna_nunca_chega_ao_critico_e_perigo_quase_sempre_chega(tipo):
    rng, crit = np.random.default_rng(2), S["perfis"][tipo]["critico"]
    benignas = [gerar_serie(tipo, P, rng, False)[2]["verdadeiro"].max() for _ in range(100)]
    perigosas = [gerar_serie(tipo, P, rng, True)[2]["verdadeiro"].max() >= crit for _ in range(100)]
    assert max(benignas) < crit
    assert np.mean(perigosas) >= 0.9


def test_persistencia_ignora_pico_isolado():
    x = np.array([10, 10, 200, 10, 10, 10.0])
    assert indice_de_alerta(x, 120, persistencia=3) is None
    assert indice_de_alerta(np.array([10, 130, 130, 130, 10.0]), 120, persistencia=3) == 3


@pytest.mark.parametrize("tipo", TIPOS)
def test_derivada_estimada_confere_com_a_exata_sem_ruido(tipo):
    perfil = S["perfis"][tipo]
    t, x, info = gerar_serie(tipo, P, np.random.default_rng(5), True, ruido=0.0)
    i = indice_de_alerta(x, perfil["limiar_alerta"], S["persistencia"])
    c = caracteristicas(t, x, i, perfil, S["janela"])
    centro = t[i - S["janela"] // 2 + 1] - info["t_anomalia"]       
    exata = info["r0"] * np.exp(info["k"] * centro)
    assert c["taxa"] == pytest.approx(exata, rel=0.10)


def test_prazo_ate_critico():
    assert prazo_ate_critico(100, 5, 250) == 30
    assert prazo_ate_critico(100, 0, 250, maximo=60) == 60
    assert prazo_ate_critico(300, 5, 250) == 0
    assert prazo_ate_critico(100, 0.1, 250, maximo=60) == 60


def test_regressao_logistica_discrimina(modelo):
    assert modelo.metricas["auc"] > 0.7
    rng = np.random.default_rng(99)
    p_perigo, p_benigna = [], []
    for tipo in TIPOS:
        for perigo, destino in ((True, p_perigo), (False, p_benigna)):
            for _ in range(60):
                t, x, _ = gerar_serie(tipo, P, rng, perigo)
                a = analisar_serie(tipo, t, x, P, modelo)
                destino.append(a["ocorrencia"]["probabilidade"])
    assert 0 < min(p_perigo + p_benigna) and max(p_perigo + p_benigna) < 1
    assert np.mean(p_perigo) > np.mean(p_benigna) + 0.1


def test_analisar_serie_aplica_fator_de_seguranca(modelo):
    t, x, _ = gerar_serie("fogao", P, np.random.default_rng(3), True)
    a = analisar_serie("fogao", t, x, P, modelo)
    assert a["ocorrencia"]["prazo_min"] == pytest.approx(
        a["prazo_linear_min"] * S["perfis"]["fogao"]["fator_seguranca_prazo"])
    assert analisar_serie("fogao", t[:5], x[:5], P, modelo) is None     


def test_pipeline_sensor_ate_decisao(modelo):
    rng, g = np.random.default_rng(7), Gerenciador(P)
    for tipo in TIPOS:
        t, x, _ = gerar_serie(tipo, P, rng, True)
        oc = analisar_serie(tipo, t, x, P, modelo)["ocorrencia"]
        m = g.registrar(oc["id"], oc["tipo"], oc["gravidade"], oc["probabilidade"], oc["prazo_min"])
        m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
    r = g.decidir()
    assert r["custo"] <= P["orcamento"] and r["acoes"]
    assert all(g.maquinas[i].estado is Estado.EM_RESPOSTA for i in r["acoes"])