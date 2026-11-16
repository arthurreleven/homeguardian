"""
Simulador de sensores: gera séries temporais de leituras para cada tipo de ocorrência.

Cada série tem ruído de medição e, a partir de um instante, uma anomalia de dois tipos:
- perigo:  crescimento exponencial acelerado, x(t) = x0 + (r0/k) (e^{kt} - 1);
- benigna: aquecimento que satura num patamar (ex.: panela em uso normal),
           x(t) = x0 + (patamar - x0) (1 - e^{-t/tau}).

Os parâmetros dos perfis ficam em parametros.json, seção "sensores".
"""
import numpy as np


def _anomalia_perigo(perfil: dict, rng: np.random.Generator):
    d = perfil["perigo"]
    r0 = rng.lognormal(np.log(d["r0"]), d["sigma"])
    k = rng.lognormal(np.log(d["k"]), d["sigma"])
    return (lambda s: (r0 / k) * (np.exp(k * s) - 1.0)), {"r0": float(r0), "k": float(k)}


def _anomalia_benigna(perfil: dict, rng: np.random.Generator):
    d = perfil["benigna"]
    patamar = rng.uniform(*d["plato"])
    tau = rng.lognormal(np.log(d["tau"]), d["sigma"])
    amplitude = patamar - perfil["normal"]
    return (lambda s: amplitude * (1.0 - np.exp(-s / tau))), {"patamar": float(patamar), "tau": float(tau)}


def gerar_serie(tipo: str, params: dict, rng: np.random.Generator, perigo: bool,
                duracao_min: float = 120.0, t_anomalia: float | None = None,
                ruido: float | None = None):
    """
    Devolve (t, x, info). `info` traz o rótulo (perigo), o início da anomalia e os
    parâmetros sorteados. `ruido` substitui o desvio-padrão do perfil (útil em testes).
    """
    sens = params["sensores"]
    perfil = sens["perfis"][tipo]
    t = np.arange(0.0, duracao_min, sens["passo_min"])
    t0 = float(rng.uniform(5.0, 20.0)) if t_anomalia is None else float(t_anomalia)

    f, parametros = (_anomalia_perigo if perigo else _anomalia_benigna)(perfil, rng)
    s = np.clip(t - t0, 0.0, None)
    verdadeiro = perfil["normal"] + f(s)
    sigma = perfil["ruido"] if ruido is None else ruido
    x = verdadeiro + rng.normal(0.0, sigma, size=t.shape)

    info = {"tipo": tipo, "perigo": bool(perigo), "t_anomalia": t0,
            "verdadeiro": verdadeiro, **parametros}
    return t, x, info