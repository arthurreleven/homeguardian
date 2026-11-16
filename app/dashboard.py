"""
Home Guardian: ponto de entrada do app.

Na raiz do projeto (PowerShell):
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
for caminho in (AQUI, AQUI.parent / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

import streamlit as st

st.set_page_config(page_title="Home Guardian", page_icon="🏠", layout="wide")

import tela_morador
import tela_tecnica

pagina = st.sidebar.radio("Tela", ["🏠 Morador", "📊 Técnica"])
st.sidebar.caption("Os dados são sintéticos. Os parâmetros dos sensores são hipóteses de simulação.")

if pagina.startswith("🏠"):
    tela_morador.render()
else:
    tela_tecnica.render()