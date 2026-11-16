"""
Simulação de Monte Carlo: sorteia instâncias (ocorrências, gravidade,
probabilidade, prazo, orçamento e atenção) e compara as três estratégias.

Uso (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.simulacao.monte_carlo
    python -m home_guardian.simulacao.monte_carlo --n 2000 --semente 7 --csv data/mc.csv
"""
import argparse
import csv

import numpy as np

from ..config import carregar_parametros
from ..modelo.heuristica import gulosa, ingenua
from ..modelo.otimizador import resolver

ESTRATEGIAS = (("ingenua", ingenua), ("gulosa", gulosa), ("pli", resolver))
TOL = 1e-6


def sortear_instancia(rng: np.random.Generator, params: dict):
    """Devolve (ocorrencias, orcamento, capacidade) sorteados."""
    mc = params["monte_carlo"]
    n = int(rng.choice(mc["n_ocorrencias"]["valores"], p=mc["n_ocorrencias"]["pesos"]))
    tipos = list(mc["tipos"])
    escolhidos = [str(t) for t in rng.choice(tipos, size=min(n, len(tipos)), replace=False)]

    ocorrencias = []
    for t in escolhidos:
        d = mc["tipos"][t]
        ocorrencias.append({
            "id": t, "tipo": t,
            "gravidade": int(rng.integers(d["gravidade"][0], d["gravidade"][1] + 1)),
            "probabilidade": float(rng.beta(*d["beta"])),
            "prazo_min": float(rng.uniform(*d["prazo_min"])),
        })
    rng.shuffle(ocorrencias)          # ordem de chegada aleatória (afeta a ingênua)

    b_lo, b_hi = mc["orcamento"]
    k_lo, k_hi = mc["capacidade_atencao"]
    return (ocorrencias,
            int(rng.integers(b_lo, b_hi + 1)),
            int(rng.integers(k_lo, k_hi + 1)))


def executar(n: int | None = None, semente: int | None = None,
             params: dict | None = None) -> dict[str, np.ndarray]:
    params = params or carregar_parametros()
    n = params["monte_carlo"]["n"] if n is None else n
    semente = params["monte_carlo"]["semente"] if semente is None else semente
    rng = np.random.default_rng(semente)

    col = {k: [] for k in ("n_ocorrencias", "orcamento", "capacidade")}
    for nome, _ in ESTRATEGIAS:
        for m in ("z", "risco_residual", "custo"):
            col[f"{m}_{nome}"] = []

    for _ in range(n):
        ocorrencias, B, K = sortear_instancia(rng, params)
        p = {**params, "orcamento": B, "capacidade_atencao": K}
        col["n_ocorrencias"].append(len(ocorrencias))
        col["orcamento"].append(B)
        col["capacidade"].append(K)
        for nome, funcao in ESTRATEGIAS:
            r = funcao(ocorrencias, p)
            col[f"z_{nome}"].append(r["z"])
            col[f"risco_residual_{nome}"].append(r["risco_residual"])
            col[f"custo_{nome}"].append(r["custo"])
    return {k: np.array(v) for k, v in col.items()}


def resumir(res: dict[str, np.ndarray]) -> dict:
    """Estatísticas da comparação (ganho = redução percentual de Z do PLI)."""
    resumo = {"n": len(res["z_pli"]), "z_medio": {}, "residual_medio": {}, "custo_medio": {}}
    for nome, _ in ESTRATEGIAS:
        resumo["z_medio"][nome] = float(res[f"z_{nome}"].mean())
        resumo["residual_medio"][nome] = float(res[f"risco_residual_{nome}"].mean())
        resumo["custo_medio"][nome] = float(res[f"custo_{nome}"].mean())
    for base in ("ingenua", "gulosa"):
        zb, zp = res[f"z_{base}"], res["z_pli"]
        ganho = 100 * (zb - zp) / np.where(zb == 0, 1, zb)
        melhor = zp < zb - TOL
        resumo[f"vs_{base}"] = {
            "pli_melhor_pct": float(100 * melhor.mean()),
            "empate_pct": float(100 * (np.abs(zb - zp) <= TOL).mean()),
            "pli_pior_pct": float(100 * (zp > zb + TOL).mean()),       # deve ser 0
            "ganho_medio_pct": float(ganho.mean()),
            "ganho_mediano_pct": float(np.median(ganho)),
            "ganho_p95_pct": float(np.percentile(ganho, 95)),
            "ganho_max_pct": float(ganho.max()),
            "ganho_medio_quando_melhor_pct": float(ganho[melhor].mean()) if melhor.any() else 0.0,
        }
    return resumo


def formatar_resumo(s: dict) -> str:
    l = [f"Monte Carlo: {s['n']} instâncias", "",
         f"{'Estratégia':<10} {'Z médio':>8} {'Resid. médio':>13} {'Custo médio':>12}"]
    for nome, _ in ESTRATEGIAS:
        l.append(f"{nome:<10} {s['z_medio'][nome]:8.1f} {s['residual_medio'][nome]:13.1f} {s['custo_medio'][nome]:12.2f}")
    for base in ("ingenua", "gulosa"):
        v = s[f"vs_{base}"]
        l += ["", f"PLI vs {base}:",
              f"  PLI melhor em {v['pli_melhor_pct']:.1f}% | empate {v['empate_pct']:.1f}% | PLI pior {v['pli_pior_pct']:.1f}%",
              f"  ganho de Z: médio {v['ganho_medio_pct']:.1f}% | mediano {v['ganho_mediano_pct']:.1f}% | "
              f"p95 {v['ganho_p95_pct']:.1f}% | máx {v['ganho_max_pct']:.1f}%",
              f"  ganho médio nas instâncias em que o PLI vence: {v['ganho_medio_quando_melhor_pct']:.1f}%"]
    return "\n".join(l)


def salvar_csv(res: dict[str, np.ndarray], caminho: str) -> None:
    chaves = list(res)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(chaves)
        w.writerows(zip(*(res[k] for k in chaves)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--semente", type=int)
    ap.add_argument("--csv", help="caminho para salvar os resultados por instância")
    a = ap.parse_args()
    resultados = executar(a.n, a.semente)
    print(formatar_resumo(resumir(resultados)))
    if a.csv:
        salvar_csv(resultados, a.csv)
        print(f"\nResultados salvos em {a.csv}")