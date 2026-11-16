"""Valida o PuLP contra força bruta e checa as restrições."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from home_guardian.config import carregar_cenario, carregar_parametros
from home_guardian.modelo.otimizador import resolver
from home_guardian.modelo.risco import calcular_risco


def forca_bruta(ocorrencias, params, B, K):
    cat = params["acoes"]
    ids, opcoes, risco, prob = [], [], {}, {}
    for oc in ocorrencias:
        i = oc["id"]
        ids.append(i)
        prob[i] = oc["probabilidade"]
        risco[i] = calcular_risco(oc["gravidade"], oc["probabilidade"],
                                  oc["prazo_min"], params["prazo_referencia_min"])
        viaveis = [(j, d) for j, d in params["acoes_por_tipo"][oc["tipo"]].items()
                   if cat[j]["autorizada"] and d["tempo_min"] <= oc["prazo_min"]]
        opcoes.append([None] + viaveis)
    melhor = None
    for combo in itertools.product(*opcoes):
        escolha = {i: c for i, c in zip(ids, combo) if c}
        custo = sum(d["custo"] for _, d in escolha.values())
        atencao = sum(cat[j]["atencao"] for j, _ in escolha.values())
        if custo > B or atencao > K:
            continue
        pares = {(i, j) for i, (j, _) in escolha.items()}
        if any(tuple(a) in pares and tuple(b) in pares for a, b in params["conflitos"]):
            continue
        z = sum(risco[i] * (1 - (escolha[i][1]["eficacia"] if i in escolha else 0))
                for i in ids)
        z += params["lambda_custo"] * custo
        z += params["lambda_falso_alarme"] * sum(
            (1 - prob[i]) * cat[j]["incomodo"] for i, (j, _) in escolha.items())
        if melhor is None or z < melhor:
            melhor = z
    return melhor


def test_pulp_igual_forca_bruta_c3():
    p, c = carregar_parametros(), carregar_cenario("c3")
    r = resolver(c["ocorrencias"], p)
    assert r["status"] == "Optimal"
    assert abs(r["z"] - forca_bruta(c["ocorrencias"], p, p["orcamento"], p["capacidade_atencao"])) < 1e-6


def test_restricoes_respeitadas():
    p, c = carregar_parametros(), carregar_cenario("c3")
    r = resolver(c["ocorrencias"], p)
    assert r["custo"] <= p["orcamento"]
    assert r["atencao"] <= p["capacidade_atencao"]


def test_prazo_exclui_contato_no_sobreaquecimento():
    p, c = carregar_parametros(), carregar_cenario("c3")
    r = resolver(c["ocorrencias"], p)
    assert r["acoes"].get("sobreaquecimento") != "acionar_contato"  


def test_orcamento_zero_so_acoes_gratuitas():
    p, c = carregar_parametros(), carregar_cenario("c3")
    r = resolver(c["ocorrencias"], p, orcamento=0)
    assert r["custo"] == 0