"""
Cálculo simbólico (SymPy) do modelo de crescimento e do risco acumulado.

Modelo de uma anomalia perigosa:   x(t) = x0 + (r0/k) (e^{kt} - 1)
    taxa (derivada):               dx/dt = r0 e^{kt}
    tempo até o valor crítico:     t* = ln(1 + k (x_crit - x0) / r0) / k
Risco acumulado de uma ocorrência não tratada (Seção 5 do modelo matemático):
    h(s) = S p e^{k s}   =>   Risco(t) = ∫0^t h(s) ds = S p (e^{kt} - 1) / k
"""
import sympy as sp
from scipy.integrate import quad

t, s = sp.symbols("t s", positive=True)
x0, r0, k, x_crit, S, p = sp.symbols("x0 r0 k x_crit S p", positive=True)

x_perigo = x0 + (r0 / k) * (sp.exp(k * t) - 1)
taxa_simbolica = sp.diff(x_perigo, t)                          
t_critico_simbolico = sp.simplify(sp.solve(sp.Eq(x_perigo, x_crit), t)[0])

h_simbolica = S * p * sp.exp(k * s)
risco_acumulado_simbolico = sp.simplify(sp.integrate(h_simbolica, (s, 0, t)))

_x = sp.lambdify((t, x0, r0, k), x_perigo, "numpy")
_taxa = sp.lambdify((t, r0, k), taxa_simbolica, "numpy")
_t_crit = sp.lambdify((x0, r0, k, x_crit), t_critico_simbolico, "numpy")
_risco = sp.lambdify((t, S, p, k), risco_acumulado_simbolico, "numpy")


def valor(tempo, x_inicial, taxa_inicial, constante):
    return _x(tempo, x_inicial, taxa_inicial, constante)


def taxa_exata(tempo, taxa_inicial, constante):
    return _taxa(tempo, taxa_inicial, constante)


def tempo_critico_exato(x_inicial, taxa_inicial, constante, critico):
    """Instante (min, desde o início da anomalia) em que x atinge o valor crítico."""
    return float(_t_crit(x_inicial, taxa_inicial, constante, critico))


def risco_acumulado(tempo, gravidade, probabilidade, constante):
    return float(_risco(tempo, gravidade, probabilidade, constante))


def risco_acumulado_numerico(tempo, gravidade, probabilidade, constante):
    """Mesma integral por quadratura numérica (SciPy), para conferir a forma analítica."""
    return quad(lambda u: gravidade * probabilidade * float(sp.exp(constante * u)), 0, tempo)[0]