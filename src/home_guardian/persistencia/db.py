"""
Persistência em SQLite: histórico de sessões, ocorrências, transições de estado,
decisões do otimizador e resultados das simulações.

Tabelas:
    sessoes             uma execução do sistema (ex.: uma "tarde em casa")
    ocorrencias         cada situação detectada, com probabilidades, prazo, ação, estado final
                        e as características (taxa, aceleração) usadas pela regressão
    eventos             transições da máquina de estados de cada ocorrência
    decisoes            cada rodada do otimizador (Z, risco, custo, recursos)
    decisao_acoes       ações escolhidas em cada decisão
    resultados_cenarios saída das simulações de cenários (para a tela técnica)

Demonstração (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.persistencia.db
"""
import sqlite3
from pathlib import Path

import numpy as np

from ..config import RAIZ

CAMINHO_PADRAO = RAIZ / "data" / "home_guardian.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS sessoes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    descricao TEXT,
    criada_em TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ocorrencias (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    sessao_id             INTEGER NOT NULL REFERENCES sessoes(id) ON DELETE CASCADE,
    ident                 TEXT    NOT NULL,
    tipo                  TEXT    NOT NULL,
    gravidade             REAL    NOT NULL,
    probabilidade_inicial REAL    NOT NULL,
    probabilidade_final   REAL    NOT NULL,
    prazo_min             REAL    NOT NULL,
    ultima_resposta       TEXT,
    acao                  TEXT,
    estado_final          TEXT    NOT NULL,
    acao_passiva          TEXT,
    taxa                  REAL,
    aceleracao            REAL,
    taxa_norm             REAL,
    acel_norm             REAL,
    perigo_real           INTEGER,
    criada_em             TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (sessao_id, ident)
);

CREATE TABLE IF NOT EXISTS eventos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ocorrencia_id INTEGER NOT NULL REFERENCES ocorrencias(id) ON DELETE CASCADE,
    ordem         INTEGER NOT NULL,
    estado_de     TEXT    NOT NULL,
    evento        TEXT    NOT NULL,
    estado_para   TEXT    NOT NULL,
    probabilidade REAL,
    UNIQUE (ocorrencia_id, ordem)
);

CREATE TABLE IF NOT EXISTS decisoes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    sessao_id      INTEGER NOT NULL REFERENCES sessoes(id) ON DELETE CASCADE,
    estrategia     TEXT    NOT NULL,
    z              REAL,
    risco_inicial  REAL,
    risco_residual REAL,
    custo          REAL,
    atencao        REAL,
    orcamento      REAL,
    capacidade     REAL,
    criada_em      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS decisao_acoes (
    decisao_id       INTEGER NOT NULL REFERENCES decisoes(id) ON DELETE CASCADE,
    ocorrencia_ident TEXT    NOT NULL,
    acao             TEXT    NOT NULL,
    PRIMARY KEY (decisao_id, ocorrencia_ident)
);

CREATE TABLE IF NOT EXISTS resultados_cenarios (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cenario        TEXT NOT NULL,
    estrategia     TEXT NOT NULL,
    z              REAL,
    risco_residual REAL,
    custo          REAL,
    atencao        REAL,
    criada_em      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ocorrencias_tipo   ON ocorrencias(tipo);
CREATE INDEX IF NOT EXISTS idx_ocorrencias_sessao ON ocorrencias(sessao_id);
"""


class Banco:
    """Acesso ao banco. Use caminho=":memory:" para testes."""

    def __init__(self, caminho: str | Path | None = None):
        caminho = CAMINHO_PADRAO if caminho is None else caminho
        if str(caminho) != ":memory:":
            Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(caminho))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(ESQUEMA)

    def fechar(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()

    def _consultar(self, sql: str, args: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def criar_sessao(self, descricao: str = "") -> int:
        with self.conn:
            return self.conn.execute("INSERT INTO sessoes (descricao) VALUES (?)", (descricao,)).lastrowid

    def salvar_maquina(self, sessao_id: int, m, taxa: float | None = None,
                       aceleracao: float | None = None, taxa_norm: float | None = None,
                       acel_norm: float | None = None, perigo_real: bool | None = None) -> int:
        """Grava (ou atualiza) uma ocorrência e seu histórico de transições."""
        rotulo = None if perigo_real is None else int(perigo_real)
        with self.conn:
            self.conn.execute(
                """INSERT INTO ocorrencias (sessao_id, ident, tipo, gravidade, probabilidade_inicial,
                       probabilidade_final, prazo_min, ultima_resposta, acao, estado_final,
                       acao_passiva, taxa, aceleracao, taxa_norm, acel_norm, perigo_real)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(sessao_id, ident) DO UPDATE SET
                       probabilidade_final = excluded.probabilidade_final,
                       prazo_min           = excluded.prazo_min,
                       ultima_resposta     = excluded.ultima_resposta,
                       acao                = excluded.acao,
                       estado_final        = excluded.estado_final,
                       acao_passiva        = excluded.acao_passiva,
                       taxa                = COALESCE(excluded.taxa, taxa),
                       aceleracao          = COALESCE(excluded.aceleracao, aceleracao),
                       taxa_norm           = COALESCE(excluded.taxa_norm, taxa_norm),
                       acel_norm           = COALESCE(excluded.acel_norm, acel_norm),
                       perigo_real         = COALESCE(excluded.perigo_real, perigo_real)""",
                (sessao_id, m.id, m.tipo, m.gravidade, m.probabilidade_inicial, m.probabilidade,
                 m.prazo_min, m.ultima_resposta, m.acao, m.estado.value,
                 getattr(m, "acao_passiva", None), taxa, aceleracao, taxa_norm, acel_norm, rotulo))
            oc_id = self.conn.execute(
                "SELECT id FROM ocorrencias WHERE sessao_id = ? AND ident = ?", (sessao_id, m.id)).fetchone()["id"]
            self.conn.execute("DELETE FROM eventos WHERE ocorrencia_id = ?", (oc_id,))
            self.conn.executemany(
                "INSERT INTO eventos (ocorrencia_id, ordem, estado_de, evento, estado_para, probabilidade) "
                "VALUES (?,?,?,?,?,?)",
                [(oc_id, n, h["de"].value, h["evento"].value, h["para"].value, h["probabilidade"])
                 for n, h in enumerate(m.historico, start=1)])
        return oc_id

    def registrar_decisao(self, sessao_id: int, resultado: dict, estrategia: str = "pli") -> int:
        """Grava uma rodada do otimizador (saída de otimizador.resolver ou Gerenciador.decidir)."""
        with self.conn:
            did = self.conn.execute(
                """INSERT INTO decisoes (sessao_id, estrategia, z, risco_inicial, risco_residual,
                       custo, atencao, orcamento, capacidade) VALUES (?,?,?,?,?,?,?,?,?)""",
                (sessao_id, estrategia, resultado["z"], resultado["risco_inicial"],
                 resultado["risco_residual"], resultado["custo"], resultado["atencao"],
                 resultado["orcamento"], resultado["capacidade"])).lastrowid
            self.conn.executemany(
                "INSERT INTO decisao_acoes (decisao_id, ocorrencia_ident, acao) VALUES (?,?,?)",
                [(did, i, a) for i, a in resultado["acoes"].items()])
        return did

    def salvar_gerenciador(self, sessao_id: int, gerenciador, resultado: dict | None = None,
                           extras: dict[str, dict] | None = None) -> dict:
        """Snapshot das ocorrências. `extras` = {id: {taxa, aceleracao, taxa_norm, acel_norm, perigo_real}}."""
        extras = extras or {}
        ids = {m.id: self.salvar_maquina(sessao_id, m, **extras.get(m.id, {}))
               for m in gerenciador.maquinas.values()}
        did = self.registrar_decisao(sessao_id, resultado) if resultado else None
        return {"ocorrencias": ids, "decisao": did}

    def registrar_cenarios(self, linhas: list[dict]) -> int:
        """Grava a saída de simulacao.cenarios.executar_todos."""
        with self.conn:
            self.conn.executemany(
                "INSERT INTO resultados_cenarios (cenario, estrategia, z, risco_residual, custo, atencao) "
                "VALUES (?,?,?,?,?,?)",
                [(l["cenario"], l["estrategia"], l["z"], l["risco_residual"], l["custo"], l["atencao"])
                 for l in linhas])
        return len(linhas)

    def limpar_cenarios(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM resultados_cenarios")

    def ultimas_ocorrencias(self, limite: int = 20, tipo: str | None = None) -> list[dict]:
        sql = "SELECT * FROM ocorrencias" + (" WHERE tipo = ?" if tipo else "") + " ORDER BY id DESC LIMIT ?"
        return self._consultar(sql, ((tipo, limite) if tipo else (limite,)))

    def eventos_da_ocorrencia(self, ocorrencia_id: int) -> list[dict]:
        return self._consultar("SELECT * FROM eventos WHERE ocorrencia_id = ? ORDER BY ordem", (ocorrencia_id,))

    def decisoes_da_sessao(self, sessao_id: int) -> list[dict]:
        decisoes = self._consultar("SELECT * FROM decisoes WHERE sessao_id = ? ORDER BY id", (sessao_id,))
        for d in decisoes:
            d["acoes"] = {r["ocorrencia_ident"]: r["acao"] for r in self._consultar(
                "SELECT ocorrencia_ident, acao FROM decisao_acoes WHERE decisao_id = ?", (d["id"],))}
        return decisoes

    def resumo_por_tipo(self) -> list[dict]:
        return self._consultar(
            """SELECT tipo,
                      COUNT(*)                                           AS n,
                      AVG(probabilidade_inicial)                         AS p_inicial_media,
                      AVG(probabilidade_final)                           AS p_final_media,
                      SUM(estado_final = 'SUPRIMIDO')                    AS suprimidas,
                      SUM(estado_final IN ('EM_RESPOSTA','RESOLVIDO'))   AS com_acao
               FROM ocorrencias GROUP BY tipo ORDER BY n DESC""")

    def media_por_estrategia(self) -> list[dict]:
        return self._consultar(
            """SELECT estrategia, COUNT(*) AS n, AVG(z) AS z_medio, AVG(risco_residual) AS residual_medio,
                      AVG(custo) AS custo_medio
               FROM resultados_cenarios GROUP BY estrategia ORDER BY z_medio""")

    def dados_para_regressao(self) -> tuple[np.ndarray, np.ndarray]:
        """(taxa_norm, acel_norm) e rótulo (perigo real) das ocorrências rotuladas."""
        linhas = self._consultar(
            "SELECT taxa_norm, acel_norm, perigo_real FROM ocorrencias "
            "WHERE perigo_real IS NOT NULL AND taxa_norm IS NOT NULL AND acel_norm IS NOT NULL ORDER BY id")
        X = np.array([[l["taxa_norm"], l["acel_norm"]] for l in linhas], dtype=float).reshape(-1, 2)
        y = np.array([l["perigo_real"] for l in linhas], dtype=int)
        return X, y

    def retreinar_do_banco(self):
        """Treina de novo a regressão logística com o histórico guardado no banco."""
        from ..sensores.deteccao import treinar_de_dados
        X, y = self.dados_para_regressao()
        return treinar_de_dados(X, y)

    def gerar_historico(self, params: dict, modelo, n: int = 150, semente: int = 11) -> int:
        """Simula n ocorrências rotuladas e grava (taxa, aceleração, rótulo) numa sessão própria."""
        from ..modelo.estados import Maquina
        from ..sensores.deteccao import analisar_serie
        from ..sensores.simulador import gerar_serie

        sens = params["sensores"]
        rng = np.random.default_rng(semente)
        tipos = list(sens["perfis"])
        sessao = self.criar_sessao(f"histórico sintético (n={n}, semente={semente})")
        gravadas = 0
        for k in range(n):
            tipo = tipos[k % len(tipos)]
            perigo = bool(rng.random() < sens["prevalencia_perigo"])
            t, x, _ = gerar_serie(tipo, params, rng, perigo)
            a = analisar_serie(tipo, t, x, params, modelo)
            if a is None:                    
                continue
            oc, c = a["ocorrencia"], a["caracteristicas"]
            m = Maquina(f"{tipo}_{k}", tipo, oc["gravidade"], oc["probabilidade"], oc["prazo_min"], params)
            self.salvar_maquina(sessao, m, taxa=c["taxa"], aceleracao=c["aceleracao"],
                                taxa_norm=c["taxa_norm"], acel_norm=c["acel_norm"], perigo_real=perigo)
            gravadas += 1
        return gravadas


def _demo():
    from ..config import carregar_parametros
    from ..modelo.estados import Gerenciador
    from ..sensores.deteccao import analisar_serie, treinar_modelo
    from ..sensores.simulador import gerar_serie
    from ..simulacao.cenarios import executar_todos

    params = carregar_parametros()
    modelo = treinar_modelo(params, n_por_tipo=500)
    rng = np.random.default_rng(21)

    with Banco() as banco:
        n = banco.gerar_historico(params, modelo, n=300)
        print(f"Histórico sintético gravado: {n} ocorrências rotuladas")
        novo = banco.retreinar_do_banco()
        print(f"Modelo original:   AUC {modelo.metricas['auc']:.2f} | acurácia {modelo.metricas['acuracia']:.2f}")
        print(f"Retreinado do SQL: AUC {novo.metricas['auc']:.2f} | acurácia {novo.metricas['acuracia']:.2f}\n")

        sessao = banco.criar_sessao("Demonstração: três ocorrências, morador sem resposta")
        g = Gerenciador(params)
        extras = {}
        for tipo in params["sensores"]["perfis"]:
            t, x, info = gerar_serie(tipo, params, rng, perigo=True)
            a = analisar_serie(tipo, t, x, params, modelo)
            oc, c = a["ocorrencia"], a["caracteristicas"]
            m = g.registrar(oc["id"], oc["tipo"], oc["gravidade"], oc["probabilidade"], oc["prazo_min"])
            m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
            extras[oc["id"]] = {"taxa": c["taxa"], "aceleracao": c["aceleracao"],
                                "taxa_norm": c["taxa_norm"], "acel_norm": c["acel_norm"],
                                "perigo_real": info["perigo"]}
        banco.salvar_gerenciador(sessao, g, g.decidir(), extras)
        banco.limpar_cenarios()
        banco.registrar_cenarios(executar_todos())

        print(f"Banco: {CAMINHO_PADRAO}\n")
        print("Ocorrências da demonstração:")
        for o in banco.ultimas_ocorrencias(3):
            print(f"  #{o['id']} {o['tipo']:<17} p {o['probabilidade_inicial']:.2f}->{o['probabilidade_final']:.2f} "
                  f"estado {o['estado_final']:<11} ação {o['acao']} | passiva {o['acao_passiva']}")
        print("\nZ médio por estratégia (cenários):")
        for r in banco.media_por_estrategia():
            print(f"  {r['estrategia']:<8} {r['z_medio']:.1f}")


if __name__ == "__main__":
    _demo()