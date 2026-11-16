"""
Estratégias de comparação para o otimizador:

- ingenua: trata as ocorrências por ordem de chegada e aplica a ação mais
  eficaz que ainda couber nos recursos.
- gulosa: ordena todas as (ocorrência, ação) pelo ganho líquido por custo e
  vai escolhendo enquanto os recursos permitirem.

Ambas respeitam as mesmas restrições do PLI (R1 a R6) e devolvem o mesmo
formato de resultado de otimizador.resolver, para comparação direta.
"""
from .risco import calcular_risco


def _preparar(ocorrencias, params, orcamento, capacidade):
    catalogo = params["acoes"]
    ctx = {
        "B": params["orcamento"] if orcamento is None else orcamento,
        "K": params["capacidade_atencao"] if capacidade is None else capacidade,
        "lam_c": params["lambda_custo"],
        "lam_f": params["lambda_falso_alarme"],
        "catalogo": catalogo,
        "conflitos": [(tuple(a), tuple(b)) for a, b in params.get("conflitos", [])],
        "risco": {}, "prob": {}, "candidatas": {}, "ordem": [],
    }
    ordenadas = sorted(enumerate(ocorrencias),
                       key=lambda t: (t[1].get("chegada_min", t[0]), t[0]))
    for _, oc in ordenadas:
        i = oc["id"]
        ctx["ordem"].append(i)
        ctx["risco"][i] = oc.get("risco") or calcular_risco(
            oc["gravidade"], oc["probabilidade"], oc["prazo_min"],
            params["prazo_referencia_min"])
        ctx["prob"][i] = oc["probabilidade"]
        ctx["candidatas"][i] = {
            j: d for j, d in params["acoes_por_tipo"][oc["tipo"]].items()
            if catalogo[j]["autorizada"] and d["tempo_min"] <= oc["prazo_min"]
        }
    return ctx


def _cabe(ctx, escolha, i, j):
    """Verifica R1, R2, R4 e R5 ao adicionar a ação j na ocorrência i."""
    if i in escolha:
        return False
    custo = sum(ctx["candidatas"][a][b]["custo"] for a, b in escolha.items())
    atencao = sum(ctx["catalogo"][b]["atencao"] for b in escolha.values())
    if custo + ctx["candidatas"][i][j]["custo"] > ctx["B"]:
        return False
    if atencao + ctx["catalogo"][j]["atencao"] > ctx["K"]:
        return False
    pares = set(escolha.items())
    for a, b in ctx["conflitos"]:
        if ((i, j) == a and b in pares) or ((i, j) == b and a in pares):
            return False
    return True


def _resultado(ctx, escolha, nome):
    risco, cand, cat = ctx["risco"], ctx["candidatas"], ctx["catalogo"]
    residual = {i: risco[i] * (1 - (cand[i][escolha[i]]["eficacia"] if i in escolha else 0))
                for i in risco}
    custo = sum(cand[i][j]["custo"] for i, j in escolha.items())
    falso = sum((1 - ctx["prob"][i]) * cat[j]["incomodo"] for i, j in escolha.items())
    return {
        "estrategia": nome,
        "status": "Heuristica",
        "z": sum(residual.values()) + ctx["lam_c"] * custo + ctx["lam_f"] * falso,
        "acoes": dict(escolha),
        "risco_inicial": sum(risco.values()),
        "risco_residual": sum(residual.values()),
        "risco_por_ocorrencia": residual,
        "custo": custo,
        "atencao": sum(cat[j]["atencao"] for j in escolha.values()),
        "orcamento": ctx["B"],
        "capacidade": ctx["K"],
    }


def ingenua(ocorrencias, params, orcamento=None, capacidade=None):
    """Ordem de chegada; em cada ocorrência, a ação mais eficaz que couber."""
    ctx = _preparar(ocorrencias, params, orcamento, capacidade)
    escolha = {}
    for i in ctx["ordem"]:
        por_eficacia = sorted(ctx["candidatas"][i].items(),
                              key=lambda kv: kv[1]["eficacia"], reverse=True)
        for j, _ in por_eficacia:
            if _cabe(ctx, escolha, i, j):
                escolha[i] = j
                break
    return _resultado(ctx, escolha, "ingenua")


def gulosa(ocorrencias, params, orcamento=None, capacidade=None, criterio="ganho"):
    """
    Escolhe (i, j) enquanto couber, em ordem decrescente de:
    - criterio="ganho": ganho líquido (redução de risco - custo - falso alarme);
    - criterio="razao": ganho líquido / (1 + custo).
    """
    ctx = _preparar(ocorrencias, params, orcamento, capacidade)
    pares = []
    for i, acoes in ctx["candidatas"].items():
        for j, d in acoes.items():
            ganho = (ctx["risco"][i] * d["eficacia"]
                     - ctx["lam_c"] * d["custo"]
                     - ctx["lam_f"] * (1 - ctx["prob"][i]) * ctx["catalogo"][j]["incomodo"])
            if ganho > 0:
                nota = ganho if criterio == "ganho" else ganho / (1 + d["custo"])
                pares.append((nota, i, j))
    pares.sort(reverse=True)
    escolha = {}
    for _, i, j in pares:
        if _cabe(ctx, escolha, i, j):
            escolha[i] = j
    return _resultado(ctx, escolha, f"gulosa_{criterio}")