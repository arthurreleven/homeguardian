"""Persistência em SQLite."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_parametros
from home_guardian.modelo.estados import Gerenciador
from home_guardian.persistencia.db import Banco
from home_guardian.sensores.deteccao import treinar_modelo
from home_guardian.simulacao.cenarios import executar_cenario

P = carregar_parametros()


def gerenciador_com_decisao():
    g = Gerenciador(P)
    for args in (("fogao", "fogao", 9, 0.7, 12), ("vazamento", "vazamento", 6, 0.8, 20),
                 ("sobreaquecimento", "sobreaquecimento", 8, 0.6, 9)):
        m = g.registrar(*args)
        m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
    return g, g.decidir()


def test_esquema_criado():
    with Banco(":memory:") as b:
        tabelas = {r["name"] for r in b._consultar("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessoes", "ocorrencias", "eventos", "decisoes", "decisao_acoes", "resultados_cenarios"} <= tabelas


def test_salvar_gerenciador_grava_ocorrencias_eventos_e_decisao():
    g, r = gerenciador_com_decisao()
    with Banco(":memory:") as b:
        s = b.criar_sessao("teste")
        ids = b.salvar_gerenciador(s, g, r)
        assert len(b.ultimas_ocorrencias()) == 3
        oc = next(o for o in b.ultimas_ocorrencias() if o["ident"] == "fogao")
        assert oc["estado_final"] == "EM_RESPOSTA" and oc["acao"] == "cortar_gas"
        assert len(b.eventos_da_ocorrencia(ids["ocorrencias"]["fogao"])) == 4
        d = b.decisoes_da_sessao(s)[0]
        assert d["acoes"] == r["acoes"] and d["custo"] == r["custo"]


def test_salvar_duas_vezes_nao_duplica():
    g, r = gerenciador_com_decisao()
    with Banco(":memory:") as b:
        s = b.criar_sessao()
        b.salvar_gerenciador(s, g)
        b.salvar_gerenciador(s, g)
        assert len(b.ultimas_ocorrencias()) == 3
        assert len(b._consultar("SELECT * FROM eventos")) == sum(len(m.historico) for m in g.maquinas.values())


def test_atualizacao_reflete_novo_estado():
    g = Gerenciador(P)
    m = g.registrar("fogao", "fogao", 9, 0.5, 12)
    m.leitura_acima_limiar(); m.enviar_pergunta()
    with Banco(":memory:") as b:
        s = b.criar_sessao()
        b.salvar_maquina(s, m)
        assert b.ultimas_ocorrencias()[0]["estado_final"] == "CONSULTA"
        m.responder("sim")
        b.salvar_maquina(s, m)
        o = b.ultimas_ocorrencias()[0]
        assert o["estado_final"] == "SUPRIMIDO" and o["ultima_resposta"] == "sim"
        assert o["probabilidade_final"] < o["probabilidade_inicial"]


def test_apagar_sessao_remove_tudo_em_cascata():
    g, r = gerenciador_com_decisao()
    with Banco(":memory:") as b:
        s = b.criar_sessao()
        b.salvar_gerenciador(s, g, r)
        with b.conn:
            b.conn.execute("DELETE FROM sessoes WHERE id = ?", (s,))
        for tabela in ("ocorrencias", "eventos", "decisoes", "decisao_acoes"):
            assert b._consultar(f"SELECT * FROM {tabela}") == []


def test_chave_estrangeira_exige_sessao_existente():
    g, _ = gerenciador_com_decisao()
    with Banco(":memory:") as b:
        with pytest.raises(sqlite3.IntegrityError):
            b.salvar_maquina(999, next(iter(g.maquinas.values())))


def test_resumos_e_cenarios():
    g, r = gerenciador_com_decisao()
    with Banco(":memory:") as b:
        b.salvar_gerenciador(b.criar_sessao(), g, r)
        resumo = {x["tipo"]: x for x in b.resumo_por_tipo()}
        assert resumo["fogao"]["n"] == 1 and resumo["fogao"]["com_acao"] == 1
        assert b.registrar_cenarios(executar_cenario("c3")) == 3
        z = {x["estrategia"]: x["z_medio"] for x in b.media_por_estrategia()}
        assert z["pli"] <= z["ingenua"]


def test_dados_persistem_em_arquivo(tmp_path):
    g, r = gerenciador_com_decisao()
    caminho = tmp_path / "pasta_nova" / "hg.db"
    with Banco(caminho) as b:
        b.salvar_gerenciador(b.criar_sessao("arquivo"), g, r)
    with Banco(caminho) as b:                      
        assert len(b.ultimas_ocorrencias()) == 3


def test_decisao_sem_acoes():
    resultado = {"z": 5.0, "risco_inicial": 5.0, "risco_residual": 5.0, "custo": 0.0,
                 "atencao": 0.0, "orcamento": 10, "capacidade": 3, "acoes": {}}
    with Banco(":memory:") as b:
        s = b.criar_sessao()
        b.registrar_decisao(s, resultado)
        assert b.decisoes_da_sessao(s)[0]["acoes"] == {}


def test_limpar_cenarios_evita_acumulo():
    with Banco(":memory:") as b:
        b.registrar_cenarios(executar_cenario("c3"))
        b.limpar_cenarios()
        b.registrar_cenarios(executar_cenario("c3"))
        assert len(b._consultar("SELECT * FROM resultados_cenarios")) == 3


def test_dados_para_regressao_ignora_sem_rotulo():
    g = Gerenciador(P)
    a = g.registrar("a", "fogao", 9, 0.7, 12)
    b_ = g.registrar("b", "fogao", 9, 0.3, 12)
    c = g.registrar("c", "fogao", 9, 0.5, 12)
    with Banco(":memory:") as b:
        s = b.criar_sessao()
        b.salvar_maquina(s, a, taxa_norm=0.02, acel_norm=0.001, perigo_real=True)
        b.salvar_maquina(s, b_, taxa_norm=0.004, acel_norm=0.0, perigo_real=False)
        b.salvar_maquina(s, c)                      
        X, y = b.dados_para_regressao()
        assert X.shape == (2, 2) and y.tolist() == [1, 0]


def test_reabrir_sem_features_nao_apaga_as_gravadas():
    g = Gerenciador(P)
    m = g.registrar("a", "fogao", 9, 0.7, 12)
    with Banco(":memory:") as b:
        s = b.criar_sessao()
        b.salvar_maquina(s, m, taxa_norm=0.02, acel_norm=0.001, perigo_real=True)
        b.salvar_maquina(s, m)
        X, y = b.dados_para_regressao()
        assert X.tolist() == [[0.02, 0.001]] and y.tolist() == [1]


def test_retreinar_do_banco():
    modelo = treinar_modelo(P, n_por_tipo=200)
    with Banco(":memory:") as b:
        assert b.gerar_historico(P, modelo, n=150) > 50
        novo = b.retreinar_do_banco()
        assert 0.6 < novo.metricas["auc"] <= 1.0