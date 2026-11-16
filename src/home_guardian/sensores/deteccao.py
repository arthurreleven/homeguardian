"""
Detecção de ocorrências a partir das leituras.

1. indice_de_alerta:   primeiro instante em que a leitura fica acima do limiar (com persistência).
2. caracteristicas:    nível, taxa de variação (derivada, por regressão linear na janela) e
                       aceleração (regressão quadrática).
3. prazo_ate_critico:  d_i = (x_crit - x) / (dx/dt), com fator de segurança.
4. treinar_modelo:     regressão logística que estima p_i^0 = P(perigo real | características).
                       treinar_de_dados treina a partir de (X, y) vindos de qualquer fonte (ex.: SQLite).
5. analisar_serie:     junta tudo e devolve a ocorrência no formato do otimizador.

Demonstração (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.sensores.deteccao
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from .simulador import gerar_serie


def indice_de_alerta(x: np.ndarray, limiar: float, persistencia: int = 3) -> int | None:
    """Índice do fim da primeira sequência de `persistencia` leituras acima do limiar."""
    acima = (x >= limiar).astype(int)
    soma = np.convolve(acima, np.ones(persistencia, dtype=int), mode="valid")
    achou = np.flatnonzero(soma == persistencia)
    return int(achou[0] + persistencia - 1) if achou.size else None


def caracteristicas(t: np.ndarray, x: np.ndarray, i: int, perfil: dict, janela: int) -> dict:
    """Nível, taxa (derivada) e aceleração na janela que termina no índice i."""
    lo = max(0, i - janela + 1)
    ts, xs = t[lo:i + 1] - t[i], x[lo:i + 1]
    taxa = float(np.polyfit(ts, xs, 1)[0])                
    aceleracao = float(2.0 * np.polyfit(ts, xs, 2)[0])     
    escala = perfil["critico"] - perfil["normal"]
    return {"nivel": float(np.mean(xs[-3:])), "taxa": taxa, "aceleracao": aceleracao,
            "taxa_norm": taxa / escala, "acel_norm": aceleracao / escala}


def prazo_ate_critico(nivel: float, taxa: float, critico: float, maximo: float = 60.0) -> float:
    """d = (x_crit - x) / (dx/dt), em minutos, limitado a [0, maximo]."""
    if taxa <= 1e-9:
        return maximo
    return float(min(max((critico - nivel) / taxa, 0.0), maximo))


@dataclass
class ModeloLogistico:
    media: np.ndarray
    desvio: np.ndarray
    w: np.ndarray
    b: float
    metricas: dict = field(default_factory=dict)

    def probabilidade(self, taxa_norm: float, acel_norm: float) -> float:
        z = float(((np.array([taxa_norm, acel_norm]) - self.media) / self.desvio) @ self.w + self.b)
        return float(1.0 / (1.0 + np.exp(-z)))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ajustar(X: np.ndarray, y: np.ndarray, l2: float = 1e-2):
    Xb = np.hstack([X, np.ones((len(X), 1))])

    def custo(theta):
        z = Xb @ theta
        pr = 1.0 / (1.0 + np.exp(-z))
        perda = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(theta[:-1] ** 2)
        grad = Xb.T @ (pr - y) + l2 * np.append(theta[:-1], 0.0)
        return perda, grad

    r = minimize(custo, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B")
    return r.x[:-1], float(r.x[-1])


def gerar_base(params: dict, n_por_tipo: int, semente: int):
    """Simula séries rotuladas e devolve (X, y) das características no instante do alerta."""
    sens = params["sensores"]
    rng = np.random.default_rng(semente)
    X, y = [], []
    for tipo, perfil in sens["perfis"].items():
        for _ in range(n_por_tipo):
            perigo = rng.random() < sens["prevalencia_perigo"]
            t, x, _ = gerar_serie(tipo, params, rng, perigo)
            i = indice_de_alerta(x, perfil["limiar_alerta"], sens["persistencia"])
            if i is None:
                continue
            c = caracteristicas(t, x, i, perfil, sens["janela"])
            X.append([c["taxa_norm"], c["acel_norm"]])
            y.append(int(perigo))
    return np.array(X), np.array(y)


def treinar_de_dados(X: np.ndarray, y: np.ndarray, semente: int = 42) -> ModeloLogistico:
    """Treina a regressão logística com (taxa_norm, acel_norm) e rótulos 0/1."""
    if len(y) < 20 or len(set(y.tolist())) < 2:
        raise ValueError("Poucos dados ou só uma classe: preciso de perigosos e benignos.")
    rng = np.random.default_rng(semente + 1)
    ordem = rng.permutation(len(y))
    corte = int(0.7 * len(y))
    tr, te = ordem[:corte], ordem[corte:]

    media, desvio = X[tr].mean(axis=0), X[tr].std(axis=0)
    w, b = _ajustar((X[tr] - media) / desvio, y[tr])
    modelo = ModeloLogistico(media, desvio, w, b)

    p_te = np.array([modelo.probabilidade(*x) for x in X[te]])
    modelo.metricas = {"n_treino": int(len(tr)), "n_teste": int(len(te)),
                       "prevalencia": float(y.mean()),
                       "acuracia": float(((p_te >= 0.5) == y[te]).mean()),
                       "auc": _auc(y[te], p_te)}
    return modelo


def treinar_modelo(params: dict, n_por_tipo: int = 1500, semente: int = 42) -> ModeloLogistico:
    X, y = gerar_base(params, n_por_tipo, semente)
    return treinar_de_dados(X, y, semente)


def analisar_serie(tipo: str, t: np.ndarray, x: np.ndarray, params: dict,
                   modelo: ModeloLogistico) -> dict | None:
    """Devolve a ocorrência (formato do otimizador) e os detalhes da detecção, ou None."""
    sens = params["sensores"]
    perfil = sens["perfis"][tipo]
    i = indice_de_alerta(x, perfil["limiar_alerta"], sens["persistencia"])
    if i is None:
        return None
    c = caracteristicas(t, x, i, perfil, sens["janela"])
    linear = prazo_ate_critico(c["nivel"], c["taxa"], perfil["critico"], sens["prazo_maximo_min"])
    fator = perfil.get("fator_seguranca_prazo", sens["fator_seguranca_prazo"])
    prazo = linear * fator
    return {
        "ocorrencia": {"id": tipo, "tipo": tipo, "gravidade": perfil["gravidade"],
                       "probabilidade": modelo.probabilidade(c["taxa_norm"], c["acel_norm"]),
                       "prazo_min": prazo},
        "instante_alerta_min": float(t[i]),
        "caracteristicas": c,
        "prazo_linear_min": linear,
    }


def _demo():
    from ..config import carregar_parametros
    from ..modelo.estados import Gerenciador

    params = carregar_parametros()
    modelo = treinar_modelo(params)
    m = modelo.metricas
    print(f"Regressão logística: AUC {m['auc']:.2f} | acurácia {m['acuracia']:.2f} "
          f"| {m['n_treino']} treino / {m['n_teste']} teste\n")

    rng = np.random.default_rng(7)
    g = Gerenciador(params)
    print(f"{'Ocorrência':<17} {'alerta':>7} {'taxa':>9} {'prazo':>7} {'p0':>6}")
    for tipo in params["sensores"]["perfis"]:
        t, x, _ = gerar_serie(tipo, params, rng, perigo=True)
        a = analisar_serie(tipo, t, x, params, modelo)
        oc = a["ocorrencia"]
        print(f"{tipo:<17} {a['instante_alerta_min']:6.1f}m {a['caracteristicas']['taxa']:7.2f}/m "
              f"{oc['prazo_min']:6.1f}m {oc['probabilidade']:6.2f}")
        mq = g.registrar(oc["id"], oc["tipo"], oc["gravidade"], oc["probabilidade"], oc["prazo_min"])
        mq.leitura_acima_limiar(); mq.enviar_pergunta(); mq.timeout()      
    r = g.decidir()
    print("\nDecisão do otimizador (morador sem resposta):")
    for mq in g.maquinas.values():
        print(f"  {mq.id:<17} {mq.estado.value:<11} ação: {mq.acao}")
    print(f"  custo {r['custo']} | risco residual {r['risco_residual']:.1f} (inicial {r['risco_inicial']:.1f})")


if __name__ == "__main__":
    _demo()