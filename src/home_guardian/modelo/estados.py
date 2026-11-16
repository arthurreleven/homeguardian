"""
Máquina de estados da escalada de ocorrências (Matemática Discreta).

Cada ocorrência percorre um grafo dirigido de estados; as transições são
definidas pela tabela TRANSICOES, e o grafo pode ser analisado (matriz de
adjacência, alcançabilidade, conexidade forte).

Fluxo principal:
    NORMAL -> ALERTA -> CONSULTA -> {SUPRIMIDO | ESCALADA | CRITICO}
    ESCALADA/CRITICO -> EM_RESPOSTA (otimizador escolhe a ação) -> RESOLVIDO -> NORMAL

Uso (na raiz do projeto, PowerShell):
    $env:PYTHONPATH = "src"
    python -m home_guardian.modelo.estados
"""
from collections import deque
from enum import Enum

import numpy as np

from .interacao import atualizar_probabilidade, capacidade_atencao_efetiva, deve_suprimir
from .otimizador import resolver


class Estado(str, Enum):
    NORMAL = "NORMAL"
    ALERTA = "ALERTA"
    CONSULTA = "CONSULTA"
    SUPRIMIDO = "SUPRIMIDO"
    ESCALADA = "ESCALADA"
    CRITICO = "CRITICO"
    EM_RESPOSTA = "EM_RESPOSTA"
    RESOLVIDO = "RESOLVIDO"


class Evento(str, Enum):
    LEITURA_ACIMA_LIMIAR = "leitura_acima_limiar"
    LEITURA_NORMAL = "leitura_normal"
    LEITURA_PIORA = "leitura_piora"
    PERGUNTA_ENVIADA = "pergunta_enviada"
    RESPOSTA_SIM_SUPRIME = "resposta_sim_suprime"
    RESPOSTA_SIM_PERSISTE = "resposta_sim_persiste"
    RESPOSTA_AJUDA = "resposta_ajuda"
    TIMEOUT = "timeout"
    LIMITE_CONSULTAS = "limite_consultas"
    CONTATO_SEM_RESOLUCAO = "contato_sem_resolucao"
    ACAO_SELECIONADA = "acao_selecionada"
    REARME = "rearme"


E, V = Estado, Evento
TRANSICOES: dict[tuple[Estado, Evento], Estado] = {
    (E.NORMAL, V.LEITURA_ACIMA_LIMIAR): E.ALERTA,
    (E.ALERTA, V.PERGUNTA_ENVIADA): E.CONSULTA,
    (E.ALERTA, V.LEITURA_NORMAL): E.NORMAL,
    (E.CONSULTA, V.RESPOSTA_SIM_SUPRIME): E.SUPRIMIDO,
    (E.CONSULTA, V.RESPOSTA_SIM_PERSISTE): E.ALERTA,      
    (E.CONSULTA, V.RESPOSTA_AJUDA): E.CRITICO,
    (E.CONSULTA, V.TIMEOUT): E.ESCALADA,
    (E.CONSULTA, V.LIMITE_CONSULTAS): E.ESCALADA,
    (E.CONSULTA, V.LEITURA_NORMAL): E.NORMAL,
    (E.SUPRIMIDO, V.LEITURA_PIORA): E.ALERTA,
    (E.SUPRIMIDO, V.TIMEOUT): E.ALERTA,
    (E.SUPRIMIDO, V.LEITURA_NORMAL): E.NORMAL,
    (E.ESCALADA, V.CONTATO_SEM_RESOLUCAO): E.CRITICO,
    (E.ESCALADA, V.ACAO_SELECIONADA): E.EM_RESPOSTA,
    (E.ESCALADA, V.LEITURA_NORMAL): E.NORMAL,
    (E.CRITICO, V.ACAO_SELECIONADA): E.EM_RESPOSTA,
    (E.CRITICO, V.LEITURA_NORMAL): E.RESOLVIDO,
    (E.EM_RESPOSTA, V.LEITURA_NORMAL): E.RESOLVIDO,
    (E.EM_RESPOSTA, V.LEITURA_PIORA): E.CRITICO,          
    (E.RESOLVIDO, V.REARME): E.NORMAL,
}

ESTADOS = list(Estado)
ESTADOS_DE_DECISAO = (Estado.ESCALADA, Estado.CRITICO)   


class TransicaoInvalida(Exception):
    pass


def matriz_adjacencia() -> np.ndarray:
    """A[i, j] = 1 se existe alguma transição do estado i para o estado j."""
    idx = {e: k for k, e in enumerate(ESTADOS)}
    A = np.zeros((len(ESTADOS), len(ESTADOS)), dtype=int)
    for (origem, _), destino in TRANSICOES.items():
        A[idx[origem], idx[destino]] = 1
    return A


def alcancaveis(origem: Estado) -> set[Estado]:
    """Estados alcançáveis a partir de `origem` (busca em largura)."""
    vistos, fila = {origem}, deque([origem])
    while fila:
        atual = fila.popleft()
        for (o, _), d in TRANSICOES.items():
            if o == atual and d not in vistos:
                vistos.add(d)
                fila.append(d)
    return vistos


def fortemente_conexo() -> bool:
    """Todo estado alcança todos os outros (o sistema sempre volta ao NORMAL)."""
    return all(alcancaveis(e) == set(ESTADOS) for e in ESTADOS)


def diagrama_mermaid() -> str:
    """Texto Mermaid do grafo, para colar no relatório (GitHub, Notion, mermaid.live)."""
    linhas = ["stateDiagram-v2", "    [*] --> NORMAL"]
    for (o, ev), d in TRANSICOES.items():
        linhas.append(f"    {o.value} --> {d.value}: {ev.value}")
    return "\n".join(linhas)


class Maquina:
    """Estado e histórico de uma ocorrência."""

    def __init__(self, id: str, tipo: str, gravidade: float, probabilidade: float,
                 prazo_min: float, params: dict):
        self.id, self.tipo, self.gravidade = id, tipo, gravidade
        self.prazo_min = prazo_min
        self.probabilidade_inicial = probabilidade
        self.probabilidade = probabilidade
        self.params = params
        self.estado = Estado.NORMAL
        self.n_consultas = 0
        self.ultima_resposta: str | None = None
        self.acao: str | None = None
        self.historico: list[dict] = []
        self.rodadas_sem_acao = 0
        self.acao_passiva: str | None = None

    def aplicar(self, evento: Evento) -> Estado:
        proximo = TRANSICOES.get((self.estado, evento))
        if proximo is None:
            raise TransicaoInvalida(f"{self.id}: '{evento.value}' não é válido em {self.estado.value}")
        self.historico.append({"de": self.estado, "evento": evento, "para": proximo,
                               "probabilidade": round(self.probabilidade, 3)})
        self.estado = proximo
        if proximo is Estado.NORMAL:                      
            self.n_consultas, self.ultima_resposta, self.acao = 0, None, None
            self.probabilidade = self.probabilidade_inicial
            self.rodadas_sem_acao, self.acao_passiva = 0, None
        return proximo

    def leitura_acima_limiar(self) -> Estado:
        if self.estado is Estado.NORMAL:
            return self.aplicar(Evento.LEITURA_ACIMA_LIMIAR)
        if self.estado in (Estado.SUPRIMIDO, Estado.EM_RESPOSTA):
            return self.aplicar(Evento.LEITURA_PIORA)
        return self.estado                             

    def leitura_normal(self) -> Estado:
        if self.estado is Estado.NORMAL:
            return self.estado
        if self.estado is Estado.RESOLVIDO:
            return self.aplicar(Evento.REARME)
        return self.aplicar(Evento.LEITURA_NORMAL)

    def enviar_pergunta(self) -> Estado:
        self.n_consultas += 1
        return self.aplicar(Evento.PERGUNTA_ENVIADA)

    def responder(self, resposta: str) -> Estado:
        """Resposta do morador ("sim", "ajuda") ou "sem_resposta" (timeout)."""
        if self.estado is not Estado.CONSULTA:
            raise TransicaoInvalida(f"{self.id}: não há consulta pendente (estado {self.estado.value})")
        self.probabilidade = atualizar_probabilidade(self.probabilidade, resposta, self.params)
        self.ultima_resposta = resposta
        if resposta == "ajuda":
            return self.aplicar(Evento.RESPOSTA_AJUDA)
        if resposta == "sem_resposta":
            return self.aplicar(Evento.TIMEOUT)
        if deve_suprimir(self.probabilidade, self.params):
            return self.aplicar(Evento.RESPOSTA_SIM_SUPRIME)
        if self.n_consultas >= self.params["estados"]["max_consultas"]:
            return self.aplicar(Evento.LIMITE_CONSULTAS)
        return self.aplicar(Evento.RESPOSTA_SIM_PERSISTE)

    def timeout(self) -> Estado:
        """Estouro de prazo: sem resposta na CONSULTA, ou nova checagem após SUPRIMIDO."""
        if self.estado is Estado.CONSULTA:
            return self.responder("sem_resposta")
        return self.aplicar(Evento.TIMEOUT)

    def contato_sem_resolucao(self) -> Estado:
        return self.aplicar(Evento.CONTATO_SEM_RESOLUCAO)

    def como_ocorrencia(self) -> dict:
        alfa = self.params["estados"].get("envelhecimento", 0.25)
        gravidade = min(self.gravidade * (1 + alfa * self.rodadas_sem_acao), 10.0)
        return {"id": self.id, "tipo": self.tipo, "gravidade": gravidade,
                "probabilidade": self.probabilidade, "prazo_min": self.prazo_min}


class Gerenciador:
    """Mantém as máquinas e aciona o otimizador quando há ocorrências a decidir."""

    def __init__(self, params: dict):
        self.params = params
        self.maquinas: dict[str, Maquina] = {}

    def registrar(self, id: str, tipo: str, gravidade: float, probabilidade: float,
                  prazo_min: float) -> Maquina:
        self.maquinas[id] = Maquina(id, tipo, gravidade, probabilidade, prazo_min, self.params)
        return self.maquinas[id]

    def avancar_tempo(self, minutos: float) -> None:
        limite = self.params["estados"].get("prazo_critico_min", 3.0)
        for m in self.maquinas.values():
            if m.estado is not Estado.NORMAL:
                m.prazo_min = max(m.prazo_min - minutos, 0.0)
            if m.estado is Estado.ESCALADA and m.acao is None and m.prazo_min <= limite:
                m.contato_sem_resolucao()

    def a_decidir(self) -> list[Maquina]:
        return [m for m in self.maquinas.values() if m.estado in ESTADOS_DE_DECISAO]

    def decidir(self) -> dict | None:
        """Roda o PLI sobre as ocorrências em ESCALADA/CRITICO e aplica a decisão."""
        alvo = self.a_decidir()
        if not alvo:
            return None
        respostas = {m.id: m.ultima_resposta for m in self.maquinas.values() if m.ultima_resposta}
        p = {**self.params, "capacidade_atencao": capacidade_atencao_efetiva(respostas, self.params)}
        resultado = resolver([m.como_ocorrencia() for m in alvo], p)
        for m in alvo:
            if m.id in resultado["acoes"]:
                m.acao = resultado["acoes"][m.id]
                m.aplicar(Evento.ACAO_SELECIONADA)
            else:
                m.rodadas_sem_acao += 1
                m.acao_passiva = "notificar"
        return resultado

def _demo():
    from ..config import carregar_parametros
    params = carregar_parametros()

    print("=== A) Morador confirma duas vezes que está usando o fogão ===")
    g = Gerenciador(params)
    m = g.registrar("fogao", "fogao", 9, 0.7, 12)
    m.leitura_acima_limiar(); m.enviar_pergunta(); m.responder("sim")
    m.enviar_pergunta(); m.responder("sim")
    print(" -> ".join(["NORMAL"] + [h["para"].value for h in m.historico]), f"| p final = {m.probabilidade:.3f}")

    print("\n=== B) Morador não responde; vazamento e sobreaquecimento ao mesmo tempo ===")
    g = Gerenciador(params)
    for args in (("fogao", "fogao", 9, 0.7, 12), ("vazamento", "vazamento", 6, 0.8, 20),
                 ("sobreaquecimento", "sobreaquecimento", 8, 0.6, 9)):
        m = g.registrar(*args)
        m.leitura_acima_limiar(); m.enviar_pergunta(); m.timeout()
    r = g.decidir()
    for m in g.maquinas.values():
        print(f"{m.id:<17} p={m.probabilidade:.2f}  {m.estado.value:<11} ação: {m.acao}")
    print(f"Custo {r['custo']} | risco residual {r['risco_residual']:.1f} (inicial {r['risco_inicial']:.1f})")


if __name__ == "__main__":
    _demo()