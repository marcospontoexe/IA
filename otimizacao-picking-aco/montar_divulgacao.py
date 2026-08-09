# -*- coding: utf-8 -*-
"""
Monta o material de divulgacao do projeto no LinkedIn.

Nao toca em nada da pasta "Atividade 2": as figuras sao extraidas das saidas ja
gravadas dentro do proprio notebook, e os numeros vem do resultados.json.

Produz, nesta pasta:
    fig1_rotas.png ... fig3_escalabilidade.png  - figuras recuperadas do notebook
    carrossel.pdf                               - 5 slides para post de documento
    slide1.png ... slide5.png                   - os mesmos slides avulsos

Roda com o interpretador do Anaconda (e onde esta o nbformat):

    & "C:\\Users\\marcos\\anaconda3\\python.exe" montar_divulgacao.py
"""

import base64
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

import nbformat

BASE = os.path.dirname(os.path.abspath(__file__))
ATIVIDADE = os.path.join(os.path.dirname(BASE), "Atividade 2")

# Paleta: fundo escuro destaca melhor no feed do LinkedIn
FUNDO = "#0f1c2e"
TEXTO = "#f2f5f8"
SUAVE = "#93a7bd"
AZUL = "#4d9de0"
LARANJA = "#f0932b"
CINZA = "#5a6a7d"
BRANCO = "#ffffff"

# 1080 x 1080 px - formato quadrado, o que melhor aproveita o feed
POL = 10.8
DPI = 100


# ===========================================================================
# 1. Recuperar figuras e numeros da Atividade 2 (somente leitura)
# ===========================================================================

NOMES_FIGURAS = ["fig1_rotas.png", "fig2_convergencia.png", "fig3_escalabilidade.png"]


def _extrair_do_notebook():
    """Le as imagens embutidas nas saidas do notebook, em ordem de aparicao."""
    caminho = os.path.join(ATIVIDADE, "otimizacao_picking.ipynb")
    with open(caminho, encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    brutas = []
    for celula in nb.cells:
        if celula.cell_type != "code":
            continue
        for saida in celula.get("outputs", []):
            dados = saida.get("data") or {}
            if "image/png" in dados:
                brutas.append(base64.b64decode(dados["image/png"]))

    if len(brutas) != len(NOMES_FIGURAS):
        raise RuntimeError(f"esperava {len(NOMES_FIGURAS)} figuras, achei {len(brutas)}")
    return brutas


def extrair_figuras():
    """Copia as figuras para esta pasta, sem escrever nada em "Atividade 2".

    Prefere os PNG originais gravados pelo notebook (200 dpi). Se tiverem sido
    apagados, cai para as versoes embutidas nas saidas do .ipynb (100 dpi), que
    servem, mas ficam menos nitidas depois da recompressao do LinkedIn.
    """
    originais = [os.path.join(ATIVIDADE, n) for n in NOMES_FIGURAS]
    usar_originais = all(os.path.exists(c) for c in originais)
    brutas = None if usar_originais else _extrair_do_notebook()

    for i, nome in enumerate(NOMES_FIGURAS):
        if usar_originais:
            with open(originais[i], "rb") as fh:
                conteudo = fh.read()
            origem = "original 200 dpi"
        else:
            conteudo = brutas[i]
            origem = "extraida do notebook"
        with open(os.path.join(BASE, nome), "wb") as fh:
            fh.write(conteudo)
        print(f"  {nome} ({len(conteudo)/1024:.0f} KB, {origem})")
    return NOMES_FIGURAS


def carregar_numeros():
    with open(os.path.join(ATIVIDADE, "resultados.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# 2. Blocos de desenho dos slides
# ===========================================================================

def novo_slide():
    fig = plt.figure(figsize=(POL, POL), dpi=DPI, facecolor=FUNDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(FUNDO)
    return fig, ax


def texto(ax, x, y, conteudo, tam=28, cor=TEXTO, peso="normal", larg=None,
          ha="left", espaco=1.35):
    """Escreve texto, quebrando linhas em `larg` caracteres quando informado."""
    if larg:
        conteudo = "\n".join(textwrap.wrap(conteudo, larg))
    return ax.text(x, y, conteudo, fontsize=tam, color=cor, fontweight=peso,
                   ha=ha, va="top", linespacing=espaco, transform=ax.transAxes)


def rodape(ax, numero, total=5):
    """Marca d'agua discreta com autoria e posicao no carrossel."""
    ax.text(0.07, 0.045, "Marcos Daniel Santana", fontsize=15, color=CINZA,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.93, 0.045, f"{numero}/{total}", fontsize=15, color=CINZA,
            ha="right", va="center", transform=ax.transAxes)
    ax.plot([0.07, 0.93], [0.085, 0.085], color="#1e3348", lw=1.2,
            transform=ax.transAxes)


def cartao(ax, x, y, larg, alt, cor=BRANCO):
    """Retangulo arredondado claro, para acomodar as figuras de fundo branco."""
    ax.add_patch(FancyBboxPatch(
        (x, y), larg, alt, boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor=cor, edgecolor="none", transform=ax.transAxes, zorder=1))


def imagem(fig, caminho, x, y, larg, alt):
    """Encaixa uma imagem em um sub-eixo, preservando a proporcao."""
    eixo = fig.add_axes([x, y, larg, alt])
    eixo.imshow(plt.imread(caminho))
    eixo.axis("off")
    eixo.set_zorder(2)
    return eixo


# ===========================================================================
# 3. Os cinco slides
# ===========================================================================

def slide1(r):
    fig, ax = novo_slide()
    texto(ax, 0.07, 0.90, "PESQUISA OPERACIONAL  ·  PYTHON", tam=19, cor=AZUL,
          peso="bold")
    texto(ax, 0.07, 0.82,
          "Implementei duas meta-heurísticas para otimizar um centro de distribuição.",
          tam=40, larg=32, espaco=1.3)
    texto(ax, 0.07, 0.45, "Uma delas foi\npraticamente inútil.",
          tam=52, cor=LARANJA, peso="bold", espaco=1.2)
    texto(ax, 0.07, 0.24,
          "E descobrir por quê foi a parte mais valiosa do projeto.",
          tam=27, cor=SUAVE, larg=44)
    rodape(ax, 1)
    return fig


def slide2(r):
    fig, ax = novo_slide()
    texto(ax, 0.07, 0.90, "O PROBLEMA", tam=19, cor=AZUL, peso="bold")

    texto(ax, 0.07, 0.80, "55%", tam=118, cor=LARANJA, peso="bold")
    texto(ax, 0.07, 0.56,
          "da despesa operacional de um armazém vem da separação de pedidos.",
          tam=30, larg=38, espaco=1.35)
    texto(ax, 0.07, 0.40, "De Koster, Le-Duc e Roodbergen (2007)",
          tam=17, cor=CINZA)

    texto(ax, 0.07, 0.31,
          "E a maior fatia disso é o separador simplesmente andando entre os "
          "endereços — tempo que não agrega nada ao produto.",
          tam=25, cor=SUAVE, larg=46)
    rodape(ax, 2)
    return fig


def slide3(r):
    fig, ax = novo_slide()
    texto(ax, 0.07, 0.94, "O QUE CONSTRUÍ", tam=19, cor=AZUL, peso="bold")
    texto(ax, 0.07, 0.885, "Tudo do zero, sem biblioteca de otimização pronta",
          tam=23, cor=SUAVE)

    etapas = [
        ("1", "Grafo do CD com 505 nós + Floyd–Warshall", "distâncias reais de caminhamento"),
        ("2", "Algoritmo Genético", "quais pedidos vão no mesmo lote"),
        ("3", "Colônia de Formigas (ACO)", "em que ordem visitar os endereços"),
    ]
    y = 0.805
    for num, titulo, sub in etapas:
        ax.text(0.075, y, num, fontsize=28, color=LARANJA, fontweight="bold",
                ha="left", va="top", transform=ax.transAxes)
        texto(ax, 0.135, y + 0.003, titulo, tam=25, peso="bold")
        texto(ax, 0.135, y - 0.033, sub, tam=19, cor=SUAVE)
        y -= 0.075

    # A figura das rotas e o ativo mais forte do carrossel: ocupa o maior espaco
    # possivel. A caixa respeita a proporcao 2,13:1 da imagem original.
    cartao(ax, 0.045, 0.135, 0.91, 0.43)
    imagem(fig, os.path.join(BASE, "fig1_rotas.png"), 0.055, 0.145, 0.89, 0.41)
    texto(ax, 0.5, 0.118, "Mesma lista de itens, duas estratégias de percurso",
          tam=20, cor=SUAVE, ha="center")
    rodape(ax, 3)
    return fig


def slide4(r):
    fig, ax = novo_slide()
    texto(ax, 0.07, 0.94, "O ACHADO", tam=19, cor=AZUL, peso="bold")
    texto(ax, 0.07, 0.885,
          "Não comparei só antes e depois. Rodei as 4 combinações possíveis "
          "para isolar cada peça.", tam=25, larg=48, espaco=1.3)

    # Barras desenhadas aqui mesmo, no tema escuro do slide. Mostra-se apenas o
    # ganho de cada configuracao; a linha de base e o zero do eixo.
    eixo = fig.add_axes([0.09, 0.40, 0.84, 0.36], facecolor=FUNDO)
    nomes = ["só ACO", "só AG", "AG + ACO"]
    valores = [r["t2_red_fifo_aco"], r["t2_red_ag_sshape"], r["t2_red_proposto"]]
    cores = [LARANJA, "#3c5a75", LARANJA]

    barras = eixo.bar(range(3), valores, color=cores, width=0.52)
    for b, v in zip(barras, valores):
        eixo.text(b.get_x() + b.get_width() / 2, v + 0.45,
                  f"−{v:.1f}%".replace(".", ","),
                  ha="center", fontsize=30, color=TEXTO, fontweight="bold")
    eixo.set_xticks(range(3))
    eixo.set_xticklabels(nomes, fontsize=23, color=TEXTO)
    eixo.set_yticks([])
    eixo.set_ylim(0, max(valores) * 1.28)
    eixo.tick_params(length=0, colors=TEXTO, pad=10)
    for lado in eixo.spines.values():
        lado.set_visible(False)

    texto(ax, 0.5, 0.345, "redução da distância percorrida, sobre a mesma onda de pedidos",
          tam=18, cor=CINZA, ha="center")

    texto(ax, 0.07, 0.26,
          "O ACO sozinho entregava 96% do ganho. O algoritmo genético quase "
          "não se pagava.", tam=29, peso="bold", larg=34, espaco=1.3)
    rodape(ax, 4)
    return fig


def slide5(r):
    fig, ax = novo_slide()
    texto(ax, 0.07, 0.94, "POR QUÊ", tam=19, cor=AZUL, peso="bold")

    texto(ax, 0.07, 0.865,
          "Para não rodar o ACO milhares de vezes dentro do AG, usei uma "
          "heurística barata como função de aptidão.", tam=27, larg=44, espaco=1.35)
    texto(ax, 0.07, 0.665,
          "Só que isso fez o AG otimizar um critério que o roteirizador final "
          "não usava. Ele estava mirando o alvo errado.",
          tam=27, cor=SUAVE, larg=44, espaco=1.35)

    ax.plot([0.07, 0.20], [0.505, 0.505], color=LARANJA, lw=4,
            transform=ax.transAxes)
    texto(ax, 0.07, 0.455,
          "Comparar antes e depois te dá um número. Isolar as variáveis te dá "
          "uma decisão.", tam=36, peso="bold", larg=28, espaco=1.3)

    texto(ax, 0.07, 0.20,
          "Notebook completo e executável no GitHub — link nos comentários.",
          tam=24, cor=AZUL, larg=44)
    rodape(ax, 5)
    return fig


# ===========================================================================
# 4. Montagem
# ===========================================================================

def main():
    print("Recuperando figuras do notebook...")
    extrair_figuras()
    r = carregar_numeros()

    construtores = [slide1, slide2, slide3, slide4, slide5]
    figuras = [c(r) for c in construtores]

    # PDF: formato aceito pelo LinkedIn para post de documento (carrossel)
    pdf = os.path.join(BASE, "carrossel.pdf")
    with PdfPages(pdf) as saida:
        for fig in figuras:
            saida.savefig(fig, facecolor=FUNDO)
    print("Carrossel:", pdf)

    # PNGs avulsos, caso prefira publicar como imagens soltas
    for i, fig in enumerate(figuras, start=1):
        destino = os.path.join(BASE, f"slide{i}.png")
        fig.savefig(destino, facecolor=FUNDO, dpi=DPI)
        plt.close(fig)
        print("  ", os.path.basename(destino))


if __name__ == "__main__":
    main()
