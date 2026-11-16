"""
Análise de sensibilidade do Home Guardian.

1. curva_orcamento: Z, risco residual e custo conforme o orçamento B (fronteira de Pareto).
2. varrer_pesos:    efeito de lambda_custo e lambda_falso_alarme na decisão.
3. varrer_prazos:   efeito de encurtar/alongar os prazos d_i.
4. grade_orcamento_atencao: em que faixas de (B, K) o PLI se separa das heurísticas
   (média sobre instâncias sorteadas).

Uso (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.simulacao.sensibilidade
"""
import numpy as np

from ..config import carregar_cenario, carregar_parametros
from ..modelo.heuristica import gulosa, ingenua
from ..modelo.otimizador import resolver
from .monte_carlo import sortear_instancia

ESTRATEGIAS = (("ingenua", ingenua), ("gulosa", gulosa), ("pli", resolver))
TOL = 1e-6


def _rodar(ocorrencias: list[dict], params: dict) -> dict[str, dict]:
    return {nome: f(ocorrencias, params) for nome, f in ESTRATEGIAS}


def _base(nome_cenario: str, params: dict | None):
    params = params or carregar_parametros()
    cenario = carregar_cenario(nome_cenario)
    return cenario["ocorrencias"], {**params, **cenario.get("overrides", {})}


# --------------------------------------------------------------------------- 1
def curva_orcamento(nome_cenario="c3", orcamentos=range(0, 17), params=None) -> list[dict]:
    ocorrencias, p = _base(nome_cenario, params)
    linhas = []
    for B in orcamentos:
        r = _rodar(ocorrencias, {**p, "orcamento": B})
        linha = {"orcamento": B}
        for nome, res in r.items():
            linha[f"z_{nome}"] = res["z"]
            linha[f"residual_{nome}"] = res["risco_residual"]
            linha[f"custo_{nome}"] = res["custo"]
        linha["acoes_pli"] = r["pli"]["acoes"]
        linhas.append(linha)
    return linhas


# --------------------------------------------------------------------------- 2
def varrer_pesos(parametro: str, valores, nome_cenario="c3", params=None) -> list[dict]:
    """parametro: 'lambda_custo' ou 'lambda_falso_alarme'."""
    ocorrencias, p = _base(nome_cenario, params)
    linhas = []
    for v in valores:
        r = resolver(ocorrencias, {**p, parametro: v})
        linhas.append({parametro: v, "z": r["z"], "risco_residual": r["risco_residual"],
                       "custo": r["custo"], "acoes": r["acoes"]})
    return linhas


# --------------------------------------------------------------------------- 3
def varrer_prazos(fatores=(0.5, 0.75, 1.0, 1.5, 2.0), nome_cenario="c3", params=None) -> list[dict]:
    """Multiplica o prazo de todas as ocorrências por cada fator."""
    ocorrencias, p = _base(nome_cenario, params)
    linhas = []
    for f in fatores:
        oc = [{**o, "prazo_min": o["prazo_min"] * f} for o in ocorrencias]
        r = _rodar(oc, p)
        linhas.append({"fator_prazo": f, "z_pli": r["pli"]["z"], "z_gulosa": r["gulosa"]["z"],
                       "z_ingenua": r["ingenua"]["z"], "risco_residual_pli": r["pli"]["risco_residual"],
                       "acoes_pli": r["pli"]["acoes"]})
    return linhas


# --------------------------------------------------------------------------- 4
def grade_orcamento_atencao(n_instancias=100, semente=42,
                            orcamentos=range(0, 15, 2), capacidades=range(0, 5),
                            params=None) -> dict:
    """
    Para cada par (B, K), roda as mesmas n instâncias sorteadas e mede:
    ganho médio de Z do PLI sobre cada heurística (%) e % de instâncias em que o PLI vence.
    As matrizes têm linhas = K e colunas = B.
    """
    params = params or carregar_parametros()
    rng = np.random.default_rng(semente)
    instancias = [sortear_instancia(rng, params)[0] for _ in range(n_instancias)]

    Bs, Ks = list(orcamentos), list(capacidades)
    saida = {"orcamentos": Bs, "capacidades": Ks}
    for base in ("ingenua", "gulosa"):
        saida[f"ganho_{base}"] = np.zeros((len(Ks), len(Bs)))
        saida[f"vitorias_{base}"] = np.zeros((len(Ks), len(Bs)))

    for a, K in enumerate(Ks):
        for b, B in enumerate(Bs):
            p = {**params, "orcamento": B, "capacidade_atencao": K}
            z = {nome: np.array([f(oc, p)["z"] for oc in instancias]) for nome, f in ESTRATEGIAS}
            for base in ("ingenua", "gulosa"):
                zb = z[base]
                ganho = 100 * (zb - z["pli"]) / np.where(zb == 0, 1, zb)
                saida[f"ganho_{base}"][a, b] = ganho.mean()
                saida[f"vitorias_{base}"][a, b] = 100 * (z["pli"] < zb - TOL).mean()
    return saida


# ------------------------------------------------------------------ formatação
def _matriz(g: dict, chave: str, titulo: str) -> str:
    cab = "K\\B " + "".join(f"{b:>7}" for b in g["orcamentos"])
    linhas = [titulo, cab]
    for a, K in enumerate(g["capacidades"]):
        linhas.append(f"{K:>3} " + "".join(f"{g[chave][a, b]:7.1f}" for b in range(len(g["orcamentos"]))))
    return "\n".join(linhas)


def relatorio_texto() -> str:
    out = ["=== 1. Curva de orçamento (C3): Z por estratégia e risco residual do PLI ===",
           f"{'B':>3} {'Z ing.':>7} {'Z gul.':>7} {'Z PLI':>7} {'Resid.PLI':>10} {'Custo PLI':>10}"]
    for l in curva_orcamento():
        out.append(f"{l['orcamento']:>3} {l['z_ingenua']:7.1f} {l['z_gulosa']:7.1f} {l['z_pli']:7.1f} "
                   f"{l['residual_pli']:10.1f} {l['custo_pli']:10.0f}")

    for par in ("lambda_custo", "lambda_falso_alarme"):
        vals = (0, 1, 3, 6, 10) if par == "lambda_custo" else (0, 1, 3, 6, 12)
        out += ["", f"=== 2. Sensibilidade a {par} (C3, PLI) ==="]
        for l in varrer_pesos(par, vals):
            acoes = ", ".join(f"{i}:{j}" for i, j in l["acoes"].items()) or "(nenhuma)"
            out.append(f"{par}={l[par]:>4}  Z={l['z']:6.1f}  resid={l['risco_residual']:5.1f}  custo={l['custo']:2.0f}  {acoes}")

    out += ["", "=== 3. Sensibilidade aos prazos (C3; fator multiplica d_i) ===",
            f"{'fator':>6} {'Z ing.':>7} {'Z gul.':>7} {'Z PLI':>7} {'Resid.PLI':>10}"]
    for l in varrer_prazos():
        out.append(f"{l['fator_prazo']:>6} {l['z_ingenua']:7.1f} {l['z_gulosa']:7.1f} {l['z_pli']:7.1f} {l['risco_residual_pli']:10.1f}")

    g = grade_orcamento_atencao()
    out += ["", "=== 4. Onde o PLI se separa das heurísticas (100 instâncias sorteadas) ===", "",
            _matriz(g, "ganho_gulosa", "Ganho médio de Z do PLI sobre a GULOSA (%)"), "",
            _matriz(g, "vitorias_gulosa", "% de instâncias em que o PLI vence a GULOSA"), "",
            _matriz(g, "ganho_ingenua", "Ganho médio de Z do PLI sobre a INGÊNUA (%)")]
    return "\n".join(out)


if __name__ == "__main__":
    print(relatorio_texto())