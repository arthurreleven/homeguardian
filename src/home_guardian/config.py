"""Carregamento de parâmetros e cenários a partir de data/."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def _ler(caminho: Path) -> dict:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def carregar_parametros() -> dict:
    return _ler(RAIZ / "data" / "parametros.json")


def carregar_cenario(nome: str) -> dict:
    return _ler(RAIZ / "data" / "cenarios" / f"{nome.lower()}.json")