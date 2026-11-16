"""Cálculo do risco R_i = S_i * p_i * u_i (escala 0-100)."""


def calcular_risco(gravidade: float, probabilidade: float,
                   prazo_min: float, prazo_ref_min: float = 10.0) -> float:
    """
    gravidade:     S_i, de 1 a 10
    probabilidade: p_i, de 0 a 1 (já atualizada pela resposta do morador)
    prazo_min:     d_i, minutos até o dano
    prazo_ref_min: prazo de referência; prazos menores que ele saturam a urgência em 1
    """
    urgencia = min(1.0, prazo_ref_min / max(prazo_min, 1e-6))
    return 100.0 * (gravidade / 10.0) * probabilidade * urgencia