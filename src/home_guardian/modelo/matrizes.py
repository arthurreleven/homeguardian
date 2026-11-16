"""
Formulação matricial do modelo (Álgebra Linear).

Para n ocorrências e m ações do catálogo:
    R  (n)    vetor de riscos
    E  (n×m)  eficácia das ações          C (n×m)  custos
    a  (m)    atenção exigida por ação    q (m)    incômodo por ação
    F  (n×m)  máscara de viabilidade: 1 se a ação é autorizada, aplicável e cabe no prazo
    X  (n×m)  decisão binária (x_ij)

    risco residual = R · (1 - (E ∘ X) 1)         (∘ = produto elemento a elemento)
    custo          = Σ (C ∘ X)
    falso alarme   = Σ ((1 - p) qᵀ ∘ X)
    Z = risco residual + λc · custo + λf · falso alarme

As funções conferem, em NumPy, o mesmo valor que o PuLP calcula variável a variável.
"""
import numpy as np

from .risco import calcular_risco


def construir_matrizes(ocorrencias: list[dict], params: dict) -> dict:
    cat = params["acoes"]
    acoes = list(cat)
    n, m = len(ocorrencias), len(acoes)
    M = {"ids": [o["id"] for o in ocorrencias], "acoes": acoes,
         "R": np.zeros(n), "p": np.zeros(n),
         "E": np.zeros((n, m)), "C": np.zeros((n, m)), "F": np.zeros((n, m), dtype=int),
         "a": np.array([cat[j]["atencao"] for j in acoes], dtype=float),
         "q": np.array([cat[j]["incomodo"] for j in acoes], dtype=float)}
    for i, oc in enumerate(ocorrencias):
        M["R"][i] = oc.get("risco") or calcular_risco(
            oc["gravidade"], oc["probabilidade"], oc["prazo_min"], params["prazo_referencia_min"])
        M["p"][i] = oc["probabilidade"]
        for j, nome in enumerate(acoes):
            d = params["acoes_por_tipo"][oc["tipo"]].get(nome)
            if d is None:
                continue
            M["E"][i, j], M["C"][i, j] = d["eficacia"], d["custo"]
            M["F"][i, j] = int(cat[nome]["autorizada"] and d["tempo_min"] <= oc["prazo_min"])
    return M


def matriz_de_decisao(M: dict, escolhidas: dict[str, str]) -> np.ndarray:
    """Converte {id_ocorrencia: acao} na matriz binária X."""
    X = np.zeros_like(M["E"])
    for i, nome in escolhidas.items():
        X[M["ids"].index(i), M["acoes"].index(nome)] = 1.0
    return X


def valor_objetivo(M: dict, X: np.ndarray, params: dict) -> float:
    residual = float(M["R"] @ (1.0 - (M["E"] * X).sum(axis=1)))
    custo = float((M["C"] * X).sum())
    falso = float((((1.0 - M["p"])[:, None]) * M["q"][None, :] * X).sum())
    return residual + params["lambda_custo"] * custo + params["lambda_falso_alarme"] * falso


def viavel(M: dict, X: np.ndarray, params: dict,
           orcamento: float | None = None, capacidade: float | None = None) -> bool:
    B = params["orcamento"] if orcamento is None else orcamento
    K = params["capacidade_atencao"] if capacidade is None else capacidade
    if np.any(X > M["F"]) or np.any(X.sum(axis=1) > 1):          
        return False
    if (M["C"] * X).sum() > B + 1e-9:                           
        return False
    if float((X * M["a"][None, :]).sum()) > K + 1e-9:             
        return False
    for (i1, a1), (i2, a2) in params.get("conflitos", []):      
        if i1 in M["ids"] and i2 in M["ids"]:
            if X[M["ids"].index(i1), M["acoes"].index(a1)] + X[M["ids"].index(i2), M["acoes"].index(a2)] > 1:
                return False
    return True