"""
Execução dos cenários C1 a C6 com as três estratégias (ingênua, gulosa, PLI).

Uso (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.simulacao.cenarios
"""
import time

from ..config import carregar_cenario, carregar_parametros
from ..modelo.heuristica import gulosa, ingenua
from ..modelo.interacao import aplicar_respostas, capacidade_atencao_efetiva
from ..modelo.otimizador import resolver

CENARIOS = ("c1", "c2", "c3", "c4", "c5", "c6")
ESTRATEGIAS = (("ingenua", ingenua), ("gulosa", gulosa), ("pli", resolver))


def executar_cenario(nome: str, params: dict | None = None) -> list[dict]:
    """Roda as três estratégias em um cenário e devolve uma linha por estratégia."""
    params = params or carregar_parametros()
    cenario = carregar_cenario(nome)
    respostas = cenario.get("respostas", {})
    ocorrencias = aplicar_respostas(cenario["ocorrencias"], respostas, params)   # Bayes
    params = {**params,
              "capacidade_atencao": capacidade_atencao_efetiva(respostas, params),
              **cenario.get("overrides", {})}                # ex.: orçamento reduzido

    linhas = []
    for estrategia, funcao in ESTRATEGIAS:
        t0 = time.perf_counter()
        r = funcao(ocorrencias, params)
        tempo_ms = (time.perf_counter() - t0) * 1000
        linhas.append({
            "cenario": cenario["nome"],
            "descricao": cenario["descricao"],
            "estrategia": estrategia,
            "z": r["z"],
            "risco_inicial": r["risco_inicial"],
            "risco_residual": r["risco_residual"],
            "reducao_risco_pct": 100 * (1 - r["risco_residual"] / r["risco_inicial"]),
            "custo": r["custo"],
            "atencao": r["atencao"],
            "orcamento": r["orcamento"],
            "capacidade": r["capacidade"],
            "acoes": r["acoes"],
            "tempo_ms": tempo_ms,
        })
    # ganho do PLI sobre cada estratégia, em % de Z
    z_pli = next(l["z"] for l in linhas if l["estrategia"] == "pli")
    for l in linhas:
        l["ganho_pli_pct"] = 100 * (l["z"] - z_pli) / l["z"] if l["z"] else 0.0
    return linhas


def executar_todos(nomes=CENARIOS) -> list[dict]:
    params = carregar_parametros()
    return [l for n in nomes for l in executar_cenario(n, params)]


def formatar_tabela(linhas: list[dict]) -> str:
    cab = f"{'Cen.':<4} {'Estratégia':<10} {'Z':>7} {'Resid.':>7} {'Redução':>8} {'Custo':>6} {'Atenção':>8}  Ações"
    out = [cab, "-" * len(cab)]
    for l in linhas:
        acoes = ", ".join(f"{i}:{j}" for i, j in l["acoes"].items()) or "(nenhuma)"
        out.append(f"{l['cenario']:<4} {l['estrategia']:<10} {l['z']:7.1f} {l['risco_residual']:7.1f} "
                   f"{l['reducao_risco_pct']:7.1f}% {l['custo']:6.0f} {l['atencao']:8.0f}  {acoes}")
    return "\n".join(out)


if __name__ == "__main__":
    print(formatar_tabela(executar_todos()))