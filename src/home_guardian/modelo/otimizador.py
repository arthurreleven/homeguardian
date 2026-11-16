"""
Otimizador do Home Guardian: PLI binária com PuLP.

Recebe ocorrências e parâmetros, devolve a decisão. Não depende de sensor,
banco ou interface.
"""
import pulp

from .risco import calcular_risco


def resolver(ocorrencias: list[dict], params: dict,
             orcamento: float | None = None,
             capacidade: float | None = None) -> dict:
    B = params["orcamento"] if orcamento is None else orcamento
    K = params["capacidade_atencao"] if capacidade is None else capacidade
    lam_c = params["lambda_custo"]
    lam_f = params["lambda_falso_alarme"]
    catalogo = params["acoes"]

    risco, prob_perigo, candidatas = {}, {}, {}
    for oc in ocorrencias:
        i = oc["id"]
        risco[i] = oc.get("risco") or calcular_risco(
            oc["gravidade"], oc["probabilidade"], oc["prazo_min"],
            params["prazo_referencia_min"])
        prob_perigo[i] = oc["probabilidade"]
        # R3 (prazo) e R6 (autorização) aplicados como filtro de viabilidade
        candidatas[i] = {
            j: d for j, d in params["acoes_por_tipo"][oc["tipo"]].items()
            if catalogo[j]["autorizada"] and d["tempo_min"] <= oc["prazo_min"]
        }

    modelo = pulp.LpProblem("home_guardian", pulp.LpMinimize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i, acoes in candidatas.items() for j in acoes}

    reducao = pulp.lpSum(risco[i] * candidatas[i][j]["eficacia"] * x[i, j] for i, j in x)
    custo = pulp.lpSum(candidatas[i][j]["custo"] * x[i, j] for i, j in x)
    falso = pulp.lpSum((1 - prob_perigo[i]) * catalogo[j]["incomodo"] * x[i, j] for i, j in x)
    atencao = pulp.lpSum(catalogo[j]["atencao"] * x[i, j] for i, j in x)

    modelo += sum(risco.values()) - reducao + lam_c * custo + lam_f * falso

    for i, acoes in candidatas.items():                      
        if acoes:
            modelo += pulp.lpSum(x[i, j] for j in acoes) <= 1, f"R1_{i}"
    modelo += custo <= B, "R2_orcamento"                     
    modelo += atencao <= K, "R4_atencao"                     
    for n, (a, b) in enumerate(params.get("conflitos", [])):  
        a, b = tuple(a), tuple(b)
        if a in x and b in x:
            modelo += x[a] + x[b] <= 1, f"R5_conflito_{n}"

    modelo.solve(pulp.PULP_CBC_CMD(msg=False))

    escolhidas = [(i, j) for (i, j), v in x.items() if v.value() and v.value() > 0.5]
    eficacia = {i: 0.0 for i in risco}
    for i, j in escolhidas:
        eficacia[i] = candidatas[i][j]["eficacia"]
    residual = {i: risco[i] * (1 - eficacia[i]) for i in risco}

    return {
        "status": pulp.LpStatus[modelo.status],
        "z": pulp.value(modelo.objective),
        "acoes": dict(escolhidas),
        "risco_inicial": sum(risco.values()),
        "risco_residual": sum(residual.values()),
        "risco_por_ocorrencia": residual,
        "custo": sum(candidatas[i][j]["custo"] for i, j in escolhidas),
        "atencao": sum(catalogo[j]["atencao"] for _, j in escolhidas),
        "orcamento": B,
        "capacidade": K,
    }