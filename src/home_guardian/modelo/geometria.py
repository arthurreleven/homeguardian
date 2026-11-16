"""
Geometria analítica da regressão logística.

No plano (taxa_norm, acel_norm), os pontos em que a regressão dá P(perigo) = limiar
formam uma reta  a x + b y + c = 0. A distância com sinal de um ponto a essa reta mede
"quão longe da dúvida" a leitura está.

Funciona com o ModeloLogistico de sensores/deteccao.py (atributos media, desvio, w, b).
"""
import numpy as np


def fronteira_decisao(modelo, limiar: float = 0.5) -> tuple[float, float, float]:
    """Coeficientes (a, b, c) da reta a x + b y + c = 0 onde P(perigo) = limiar."""
    w1, w2 = modelo.w
    m1, m2 = modelo.media
    s1, s2 = modelo.desvio
    logit = float(np.log(limiar / (1.0 - limiar)))
    return (float(w1 / s1), float(w2 / s2),
            float(modelo.b - w1 * m1 / s1 - w2 * m2 / s2 - logit))


def distancia_a_fronteira(modelo, taxa_norm: float, acel_norm: float, limiar: float = 0.5) -> float:
    """Distância euclidiana com sinal à reta (positiva do lado de 'perigo')."""
    a, b, c = fronteira_decisao(modelo, limiar)
    return float((a * taxa_norm + b * acel_norm + c) / np.hypot(a, b))