"""
Interação com o morador: atualização bayesiana da probabilidade de perigo real.

    P(perigo | resposta) = P(resp | perigo) p0 / [P(resp | perigo) p0 + P(resp | seguro) (1 - p0)]

Respostas possíveis (as da tela do morador):
    "sim"          -> "Sim, estou usando"
    "ajuda"        -> "Preciso de ajuda"
    "sem_resposta" -> o morador não respondeu dentro do prazo

As verossimilhanças ficam em parametros.json (seção "interacao") e são hipóteses
de simulação, a serem calibradas.
"""

RESPOSTAS = ("sim", "ajuda", "sem_resposta")


def _verossimilhanca(resposta: str, params: dict) -> tuple[float, float]:
    tabela = params["interacao"]["verossimilhanca"]
    if resposta not in tabela:
        raise ValueError(f"Resposta inválida: {resposta!r}. Use uma de {RESPOSTAS}.")
    return tabela[resposta]["perigo"], tabela[resposta]["seguro"]


def atualizar_probabilidade(p0: float, resposta: str, params: dict) -> float:
    """Probabilidade a posteriori de a ocorrência ser um perigo real."""
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 deve estar entre 0 e 1.")
    l_perigo, l_seguro = _verossimilhanca(resposta, params)
    numerador = l_perigo * p0
    denominador = numerador + l_seguro * (1.0 - p0)
    return numerador / denominador if denominador > 0 else p0


def aplicar_respostas(ocorrencias: list[dict], respostas: dict[str, str], params: dict) -> list[dict]:
    """
    Devolve novas ocorrências com `probabilidade` atualizada para as que têm resposta
    (respostas = {id_da_ocorrencia: resposta}). A original fica em `probabilidade_inicial`.
    """
    saida = []
    for oc in ocorrencias:
        novo = dict(oc)
        if oc["id"] in respostas:
            novo["probabilidade_inicial"] = oc["probabilidade"]
            novo["probabilidade"] = atualizar_probabilidade(oc["probabilidade"], respostas[oc["id"]], params)
            novo["resposta"] = respostas[oc["id"]]
        saida.append(novo)
    return saida


def capacidade_atencao_efetiva(respostas: dict[str, str], params: dict) -> float:
    """
    Se o morador foi consultado e não respondeu a nenhuma consulta, ele não está
    disponível para agir: a atenção disponível cai (padrão: 0). Caso contrário, vale a
    capacidade configurada. É uma hipótese de modelagem, ajustável em parametros.json.
    """
    if respostas and all(r == "sem_resposta" for r in respostas.values()):
        return params["interacao"]["atencao_se_nenhuma_resposta"]
    return params["capacidade_atencao"]


def deve_suprimir(probabilidade: float, params: dict) -> bool:
    """True se a probabilidade ficou abaixo do limiar e o alerta pode ser suprimido."""
    return probabilidade < params["interacao"]["limiar_supressao"]