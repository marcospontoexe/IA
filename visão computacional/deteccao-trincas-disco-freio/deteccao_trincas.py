"""
Detecção de trincas num disco de freio por processamento de imagens.

Localiza trincas térmicas na pista de um disco de freio a partir de uma foto, separa-as
dos riscos normais de desgaste e decide se o disco passa na triagem. Cada etapa trabalha
sobre a imagem gerada pela anterior:

    original -> escala de cinza -> equalização de histograma -> filtro gaussiano
    -> pista desenrolada (transformação geométrica) -> filtro de média ao longo do raio
    -> Sobel de 2ª ordem na direção da volta -> limiarização
    -> classificação das manchas por forma -> resultado de volta na foto

Todas as imagens, da original à final, são gravadas na pasta de saída, para conferência.
Documentação completa no README.md.

Uso:
    python deteccao_trincas.py                    # usa disco_freio_trincado.jpg da pasta do script
    python deteccao_trincas.py outra_foto.jpg     # analisa outra foto
    python deteccao_trincas.py --teste            # valida a sequência num disco sintético

Requer o OpenCV e o NumPy (pip install -r requirements.txt).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Parâmetros. Foram calibrados na foto do disco, de 1280 px de largura; fotos de
# outro tamanho são redimensionadas para essa largura antes da análise.
# ---------------------------------------------------------------------------
LARGURA_PADRAO = 1280      # largura de trabalho, em pixels
SIGMA_GAUSS = 1.5          # filtro gaussiano: tira o ruído fino e preserva linhas de 2 a 4 px
RAIO_INTERNO = 0.75        # pista analisada, em frações do raio da face do disco: exclui o cubo
RAIO_EXTERNO = 0.97        # e a aresta externa, cujo brilho confundiria os filtros
N_RAIOS = 120              # colunas da pista desenrolada (as linhas são calculadas: 1 por pixel de arco)
MEDIA_RAIO = 5             # filtro de média de 1 x 5 ao longo do raio
LIMIAR_SOBEL = 75          # resposta mínima do Sobel na volta para um pixel ser candidato
AREA_MINIMA = 8            # manchas menores que isso (em pixels da faixa) são ruído
COMPRIMENTO_MINIMO = 20    # trinca: ao menos 20 colunas, cerca de 1/6 da largura da pista
ALONGAMENTO_MINIMO = 3     # comprimento pelo menos 3 vezes maior que a espessura
ANGULO_MAXIMO = 30         # e no máximo 30° de desvio da direção do raio

QUADRANTES = ["direita", "baixo", "esquerda", "cima"]   # ordem em que a pista é desenrolada
VERMELHO, AMARELO, VERDE = (0, 0, 255), (0, 200, 255), (0, 255, 0)   # cores BGR do OpenCV


# ---------------------------------------------------------------------------
# Leitura e gravação. O cv2.imread e o cv2.imwrite não aceitam caminhos com
# acento no Windows (o caminho deste projeto tem "visão"), então os arquivos
# são lidos e gravados como bytes pelo NumPy e decodificados pelo OpenCV.
# ---------------------------------------------------------------------------
def ler_imagem(caminho):
    """Lê uma imagem colorida de qualquer caminho, inclusive com acentos."""
    imagem = cv2.imdecode(np.fromfile(str(caminho), dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise SystemExit(f"Não foi possível abrir a imagem: {caminho}")
    return imagem


def salvar_imagem(pasta, nome, imagem):
    """Grava a imagem em PNG (sem perdas) e devolve o caminho do arquivo."""
    caminho = Path(pasta) / nome
    ok, codificada = cv2.imencode(".png", imagem)
    if not ok:
        raise RuntimeError(f"Falha ao codificar {nome}")
    codificada.tofile(str(caminho))
    return caminho


def para_visualizacao(resposta):
    """Estica uma resposta de filtro (números reais) para 0-255, só para salvar e ver."""
    topo = np.percentile(resposta, 99.5)          # ignora os 0,5 % mais altos, para não escurecer o resto
    if topo <= 0:
        return np.zeros(resposta.shape, np.uint8)
    return np.clip(resposta / topo * 255, 0, 255).astype(np.uint8)


def quadrante_do_angulo(angulo):
    """Quadrante de um ângulo em radianos: direita de -45° a 45°, baixo de 45° a 135° etc."""
    graus = (np.degrees(angulo) + 45) % 360
    return np.minimum(3, (graus // 90).astype(int))


def faixa_para_exibicao(faixa, angulos):
    """
    Prepara a pista desenrolada para exibição: deita a faixa (raio na vertical, raio
    externo em cima) e corta em quatro pedaços, um por quadrante, empilhados na ordem
    direita, baixo, esquerda e cima.
    """
    deitada = np.swapaxes(faixa, 0, 1)[::-1]
    quad = quadrante_do_angulo(angulos)
    pedacos = [deitada[:, quad == k] for k in range(4)]
    largura = max(p.shape[1] for p in pedacos)
    blocos = []
    for p in pedacos:
        falta = largura - p.shape[1]               # completa com preto os quadrantes mais curtos
        blocos.append(np.pad(p, [(0, 0), (0, falta)] + [(0, 0)] * (p.ndim - 2)))
        blocos.append(np.full((8, largura) + p.shape[2:], 128, np.uint8))   # separador cinza
    return np.vstack(blocos[:-1])


# ---------------------------------------------------------------------------
# A pista de frenagem: na foto oblíqua, a face do disco é uma elipse.
# ---------------------------------------------------------------------------
def estimar_elipse_do_disco(cinza):
    """
    Estima a elipse da face do disco a partir da silhueta dele.

    1. A limiarização de Otsu separa o disco (escuro) do fundo (claro).
    2. Fechamento morfológico, maior objeto e envoltória convexa tapam os buracos
       da máscara (o cubo e os reflexos também são claros).
    3. Os extremos da linha mais larga da silhueta dão o centro horizontal e o
       semieixo horizontal.
    4. O centro vertical e o semieixo vertical saem de um ajuste por mínimos quadrados
       ao arco de cima da silhueta, que não sofre interferência da lateral do disco
       (visível embaixo). Só entram pontos onde a máscara do Otsu tem borda de verdade,
       e não os trechos retos que a envoltória convexa cria para tapar reflexos.
    """
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    limiar_otsu, mascara = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n < 2:
        raise SystemExit("Nenhum objeto escuro encontrado: o disco precisa estar sobre um fundo claro.")
    disco = ((rotulos == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    contornos, _ = cv2.findContours(disco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    silhueta = np.zeros_like(disco)
    cv2.drawContours(silhueta, [cv2.convexHull(max(contornos, key=cv2.contourArea))], -1, 255, cv2.FILLED)

    # Passo 3: linha mais larga
    larguras = np.array([np.ptp(np.flatnonzero(linha)) if linha.any() else 0 for linha in silhueta])
    y_largo = int(np.flatnonzero(larguras >= larguras.max() - 2)[0])
    xs = np.flatnonzero(silhueta[y_largo])
    cx, semi_a = (xs[0] + xs[-1]) / 2, (xs[-1] - xs[0]) / 2

    # Passo 4: arco de cima. Para cada coluna da parte central, o pixel mais alto da silhueta.
    nucleo = np.ones((5, 5), np.uint8)
    borda_real = cv2.dilate(disco, nucleo) & ~cv2.erode(disco, nucleo)
    colunas, topos = [], []
    for x in range(int(cx - 0.8 * semi_a), int(cx + 0.8 * semi_a) + 1):
        ys = np.flatnonzero(silhueta[:, x])
        if len(ys) and borda_real[ys[0], x]:
            colunas.append(x)
            topos.append(ys[0])
    # Na elipse, o arco de cima é y = cy - b * raiz(1 - ((x - cx) / a)²): linear em cy e b
    s = np.sqrt(1 - ((np.array(colunas) - cx) / semi_a) ** 2)
    sistema = np.column_stack([np.ones_like(s), -s])
    cy, semi_b = np.linalg.lstsq(sistema, np.array(topos, float), rcond=None)[0]
    return (float(cx), float(cy), float(semi_a), float(semi_b)), limiar_otsu, silhueta


def angulos_da_faixa(elipse):
    """
    Ângulos das linhas da faixa, a partir de -45° (assim cada quadrante fica inteiro).
    O espaçamento não é constante: cada linha cobre cerca de 1 px de arco no meio da
    pista. Com passos de ângulo iguais, os lados da elipse achatada ficariam esticados
    na faixa, e as trincas ali ficariam grossas demais para o filtro.
    """
    _, _, semi_a, semi_b = elipse
    raio_medio = (RAIO_INTERNO + RAIO_EXTERNO) / 2
    fino = np.linspace(-np.pi / 4, 7 * np.pi / 4, 20000)
    passo = raio_medio * np.sqrt((semi_a * np.sin(fino)) ** 2 + (semi_b * np.cos(fino)) ** 2)  # px de arco por radiano
    arco = np.concatenate([[0], np.cumsum((passo[1:] + passo[:-1]) / 2 * np.diff(fino))])
    n_linhas = int(round(arco[-1]))
    return np.interp(np.arange(n_linhas) + 0.5, arco, fino).astype(np.float32)


def desenrolar_pista(imagem, elipse, angulos):
    """
    Transformação geométrica polar: cada linha da faixa é um ângulo e cada coluna é um
    raio da elipse. Assim, as trincas radiais viram linhas horizontais e os riscos
    circulares de desgaste viram linhas verticais.
    """
    cx, cy, semi_a, semi_b = elipse
    raios = np.linspace(RAIO_INTERNO, RAIO_EXTERNO, N_RAIOS, dtype=np.float32)
    mapa_x = (cx + semi_a * raios[None, :] * np.cos(angulos)[:, None]).astype(np.float32)
    mapa_y = (cy + semi_b * raios[None, :] * np.sin(angulos)[:, None]).astype(np.float32)
    return cv2.remap(imagem, mapa_x, mapa_y, cv2.INTER_LINEAR)


def enrolar_de_volta(faixa, elipse, angulos, forma):
    """Faz o caminho inverso: leva cada pixel da faixa de volta à posição dele na foto."""
    cx, cy, semi_a, semi_b = elipse
    yy, xx = np.mgrid[0:forma[0], 0:forma[1]].astype(np.float32)
    u, v = (xx - cx) / semi_a, (yy - cy) / semi_b
    angulo = (np.arctan2(v, u) + np.pi / 4) % (2 * np.pi) - np.pi / 4     # mesma faixa de -45° a 315°
    linha = np.interp(angulo, angulos, np.arange(len(angulos))).astype(np.float32)
    coluna = ((np.sqrt(u ** 2 + v ** 2) - RAIO_INTERNO) / (RAIO_EXTERNO - RAIO_INTERNO)
              * (N_RAIOS - 1)).astype(np.float32)
    return cv2.remap(faixa, coluna, linha, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# Classificação das manchas da faixa limiarizada por forma.
# ---------------------------------------------------------------------------
def classificar_manchas(binaria, angulos):
    """
    Mede cada mancha (componente conectado) da faixa limiarizada. Na faixa, uma trinca
    radial é uma mancha comprida na direção do raio (das colunas). O eixo principal vem
    da covariância das coordenadas dos pixels; a tolerância de 30° absorve a perspectiva
    e trincas que não são perfeitamente radiais. Manchas curtas, redondas ou inclinadas
    demais são pontos quentes, marcas ou ruído.
    """
    n, rotulos, stats, centros = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    classes = np.zeros(binaria.shape, np.uint8)   # 0 = fundo, 1 = trinca, 2 = outra mancha
    trincas, outras = [], 0
    for i in range(1, n):                         # o rótulo 0 é o fundo
        if stats[i, cv2.CC_STAT_AREA] < AREA_MINIMA:
            continue
        x, y, w, h = stats[i, :4]
        recorte = rotulos[y:y + h, x:x + w] == i  # só a caixa da mancha: mais rápido que varrer a faixa toda
        ys, xs = np.nonzero(recorte)
        pontos = np.column_stack([xs, ys]).astype(np.float64)
        autovalores, autovetores = np.linalg.eigh(np.cov((pontos - pontos.mean(axis=0)).T))
        menor, maior = max(autovalores[0], 1e-9), max(autovalores[1], 1e-9)
        comprimento = np.sqrt(12 * maior)          # comprimento de um segmento com essa variância
        alongamento = np.sqrt(maior / menor)       # 1 = mancha redonda; grande = linha
        desvio = np.degrees(np.arccos(min(1.0, abs(autovetores[0, 1]))))   # 0° = na direção do raio
        if comprimento >= COMPRIMENTO_MINIMO and alongamento >= ALONGAMENTO_MINIMO and desvio <= ANGULO_MAXIMO:
            classes[y:y + h, x:x + w][recorte] = 1
            angulo = angulos[min(len(angulos) - 1, int(round(centros[i][1])))]
            trincas.append({"comprimento": float(comprimento), "quadrante": QUADRANTES[quadrante_do_angulo(angulo)]})
        else:
            classes[y:y + h, x:x + w][recorte] = 2
            outras += 1
    return classes, trincas, outras


def pintar_classes(classes):
    """Imagem colorida da faixa classificada: trincas em vermelho, outras manchas em amarelo."""
    cor = np.zeros(classes.shape + (3,), np.uint8)
    cor[classes == 1] = VERMELHO
    cor[classes == 2] = AMARELO
    return cor


def desenhar_contorno_da_pista(foto, elipse, espessura):
    """Desenha as duas elipses que limitam a pista analisada."""
    cx, cy, semi_a, semi_b = elipse
    for fracao in (RAIO_INTERNO, RAIO_EXTERNO):
        eixos = (int(semi_a * fracao), int(semi_b * fracao))
        cv2.ellipse(foto, (int(cx), int(cy)), eixos, 0, 0, 360, VERDE, espessura, cv2.LINE_AA)


def desenhar_resultado(original, classes_na_foto, elipse, n_trincas, reprovado):
    """Imagem final: trincas sobre a foto, limites da pista e, embaixo, a legenda."""
    foto = original.copy()
    trincas = cv2.dilate((classes_na_foto == 1).astype(np.uint8), np.ones((3, 3), np.uint8))  # traço mais visível
    foto[trincas > 0] = VERMELHO
    desenhar_contorno_da_pista(foto, elipse, 1)
    legenda = np.full((60, foto.shape[1], 3), 40, np.uint8)   # as fontes do OpenCV não têm acentos
    cv2.putText(legenda, f"Trincas radiais: {n_trincas}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, VERMELHO, 2, cv2.LINE_AA)
    texto = "Triagem: REPROVADO - encaminhar para inspecao" if reprovado else "Triagem: aprovado"
    cv2.putText(legenda, texto, (foto.shape[1] // 3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([foto, legenda])


def ampliar_trecho(original, final, classes_na_foto, largura=320, altura=200):
    """
    Recorta, lado a lado e ampliados 2 vezes, o mesmo trecho da foto original e do
    resultado: o trecho de largura x altura com mais pixels de trinca.
    """
    trincas = (classes_na_foto == 1).astype(np.float32)
    densidade = cv2.boxFilter(trincas, -1, (largura, altura), normalize=False)   # pixels de trinca na janela
    y, x = np.unravel_index(np.argmax(densidade), densidade.shape)                # centro da janela mais cheia
    y0 = int(np.clip(y - altura // 2, 0, original.shape[0] - altura))
    x0 = int(np.clip(x - largura // 2, 0, original.shape[1] - largura))
    recortes = [img[y0:y0 + altura, x0:x0 + largura] for img in (original, final)]
    sep = np.full((altura, 6, 3), 255, np.uint8)
    lado_a_lado = np.hstack([recortes[0], sep, recortes[1]])
    return cv2.resize(lado_a_lado, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), (x0, y0)


def desenhar_regiao(original, elipse):
    """Foto com a pista analisada e os quatro quadrantes em que ela é desenrolada (para conferência)."""
    foto = original.copy()
    desenhar_contorno_da_pista(foto, elipse, 2)
    cx, cy, semi_a, semi_b = elipse
    ponto = lambda fr, ang: (int(cx + semi_a * fr * np.cos(ang)), int(cy + semi_b * fr * np.sin(ang)))
    for ang in np.deg2rad([-45, 45, 135, 225]):                 # divisas entre os quadrantes
        cv2.line(foto, ponto(RAIO_INTERNO, ang), ponto(RAIO_EXTERNO, ang), VERDE, 2, cv2.LINE_AA)
    for nome, ang in zip(QUADRANTES, np.deg2rad([0, 90, 180, 270])):
        x, y = ponto(0.62, ang)                                 # nome por dentro da pista
        cv2.putText(foto, nome, (x - 45, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, VERDE, 2, cv2.LINE_AA)
    return foto


# ---------------------------------------------------------------------------
# Sequência principal: cada etapa recebe a imagem da etapa anterior.
# ---------------------------------------------------------------------------
def processar(caminho_imagem, pasta_saida):
    """Roda a sequência inteira numa foto, grava as imagens e devolve um resumo."""
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    gravadas = []

    def salvar(nome, imagem):
        gravadas.append(salvar_imagem(pasta_saida, nome, imagem))

    # Etapa 0: imagem original
    original = ler_imagem(caminho_imagem)
    salvar("00_original.png", original)
    if original.shape[1] != LARGURA_PADRAO:          # escala: deixa os parâmetros válidos para outras fotos
        fator = LARGURA_PADRAO / original.shape[1]
        original = cv2.resize(original, None, fx=fator, fy=fator, interpolation=cv2.INTER_AREA)
        salvar("aux_redimensionada.png", original)

    # Etapa 1: escala de cinza. As trincas se distinguem pelo brilho, não pela cor.
    cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    salvar("01_cinza.png", cinza)

    # Etapa 2: equalização de histograma. Espalha os tons por 0-255, realça as trincas e
    # padroniza o contraste, para que o limiar fixo da etapa 7 valha para fotos diferentes.
    equalizada = cv2.equalizeHist(cinza)
    salvar("02_equalizada.png", equalizada)

    # Etapa 3: filtro gaussiano. Tira o ruído de compressão e a textura fina do metal,
    # que o filtro de derivada da etapa 6 confundiria com trincas.
    suavizada = cv2.GaussianBlur(equalizada, (0, 0), SIGMA_GAUSS)
    salvar("03_suavizada.png", suavizada)

    # Etapa 4: transformação geométrica. A elipse do disco vem de uma limiarização de Otsu
    # sobre a imagem em cinza; a pista entre 75 % e 97 % do raio é desenrolada numa faixa.
    elipse, limiar_otsu, silhueta = estimar_elipse_do_disco(cinza)
    salvar("aux_silhueta_otsu.png", silhueta)
    salvar("aux_regiao_analisada.png", desenhar_regiao(original, elipse))
    angulos = angulos_da_faixa(elipse)
    pista = desenrolar_pista(suavizada, elipse, angulos)
    salvar("04_pista_desenrolada.png", faixa_para_exibicao(pista, angulos))

    # Etapa 5: filtro de média de 1 x 5 ao longo do raio. A trinca quase não muda (ela é
    # contínua nessa direção) e o ruído, que varia de um pixel para outro, diminui.
    media = cv2.blur(pista, (MEDIA_RAIO, 1))
    salvar("05_media_ao_longo_do_raio.png", faixa_para_exibicao(media, angulos))

    # Etapa 6: Sobel de 2ª ordem na direção da volta (linhas da faixa). Uma trinca é um vale
    # escuro atravessado pela volta, onde a segunda derivada é positiva. Os riscos circulares
    # não variam nessa direção e somem. A outra direção, a do raio, vai num arquivo auxiliar.
    sobel_volta = np.clip(cv2.Sobel(media, cv2.CV_32F, 0, 2, ksize=5), 0, None)
    salvar("06_sobel_na_volta.png", faixa_para_exibicao(para_visualizacao(sobel_volta), angulos))
    sobel_raio = np.clip(cv2.Sobel(pista, cv2.CV_32F, 2, 0, ksize=5), 0, None)
    salvar("aux_sobel_no_raio_riscos.png", faixa_para_exibicao(para_visualizacao(sobel_raio), angulos))

    # Etapa 7: limiarização. Ficam brancos só os pixels com resposta forte.
    binaria = ((sobel_volta > LIMIAR_SOBEL) * 255).astype(np.uint8)
    salvar("07_limiarizada.png", faixa_para_exibicao(binaria, angulos))

    # Etapa 8: classificação de cada mancha por forma (comprimento, alongamento e direção)
    classes, trincas, outras = classificar_manchas(binaria, angulos)
    salvar("08_classificacao.png", faixa_para_exibicao(pintar_classes(classes), angulos))

    # Etapa 9: resultado na foto, com a decisão de triagem
    classes_na_foto = enrolar_de_volta(classes, elipse, angulos, cinza.shape)
    reprovado = len(trincas) > 0                     # triagem conservadora: qualquer trinca reprova
    final = desenhar_resultado(original, classes_na_foto, elipse, len(trincas), reprovado)
    salvar("09_resultado_final.png", final)
    zoom, canto_zoom = ampliar_trecho(original, final[:original.shape[0]], classes_na_foto)
    salvar("aux_ampliacao.png", zoom)

    # Brilho médio da pista (antes da equalização) em cada quadrante, para analisar a iluminação
    pista_cinza = desenrolar_pista(cinza, elipse, angulos)
    quad = quadrante_do_angulo(angulos)
    brilho = {q: float(pista_cinza[quad == k].mean()) for k, q in enumerate(QUADRANTES)}
    return {
        "elipse": elipse,
        "limiar_otsu": float(limiar_otsu),
        "forma_faixa": binaria.shape,
        "pixels_limiarizados": int((binaria > 0).sum()),
        "trincas": trincas,
        "outras_manchas": outras,
        "trincas_quadrante": {q: sum(1 for t in trincas if t["quadrante"] == q) for q in QUADRANTES},
        "brilho_quadrante": brilho,
        "canto_ampliacao": canto_zoom,
        "reprovado": reprovado,
        "imagens": gravadas,
    }


def imprimir_resumo(resumo):
    """Mostra no console o que a sequência encontrou."""
    cx, cy, semi_a, semi_b = resumo["elipse"]
    linhas, colunas = resumo["forma_faixa"]
    comprimentos = [t["comprimento"] for t in resumo["trincas"]]
    print(f"Limiar de Otsu (disco x fundo): {resumo['limiar_otsu']:.0f}")
    print(f"Elipse da face do disco: centro ({cx:.0f}, {cy:.0f}), semieixos {semi_a:.0f} x {semi_b:.0f} px")
    print(f"Faixa desenrolada: {linhas} linhas (ângulos) x {colunas} colunas (raios)")
    print(f"Pixels da faixa acima do limiar: {resumo['pixels_limiarizados']} de {linhas * colunas}")
    print(f"Trincas radiais: {len(comprimentos)}; outras manchas descartadas: {resumo['outras_manchas']}")
    if comprimentos:
        print(f"Comprimento das trincas (colunas da faixa, de {colunas}): mínimo {min(comprimentos):.0f}, "
              f"mediana {np.median(comprimentos):.0f}, máximo {max(comprimentos):.0f}")
    print("Por quadrante (trincas | brilho médio da pista, 0-255):")
    for q in QUADRANTES:
        print(f"  {q:<9} {resumo['trincas_quadrante'][q]:>3} | {resumo['brilho_quadrante'][q]:.0f}")
    print("Triagem: REPROVADO, encaminhar para inspeção" if resumo["reprovado"] else "Triagem: aprovado")
    print(f"Trecho ampliado (canto superior esquerdo, em px): {resumo['canto_ampliacao']}")
    print("Imagens gravadas:")
    for caminho in resumo["imagens"]:
        print(f"  {caminho.name}")


# ---------------------------------------------------------------------------
# Teste: um disco sintético com trincas e riscos conhecidos.
# ---------------------------------------------------------------------------
def teste_sintetico(pasta):
    """
    Desenha um disco em perspectiva com 8 trincas radiais (2 por quadrante) cruzando
    4 riscos circulares, com luz mais forte à esquerda, como na foto. Roda a sequência e
    confere se as 8 trincas foram achadas, cada uma no quadrante certo.
    """
    pasta = Path(pasta)
    gerador = np.random.default_rng(0)
    imagem = np.full((877, 1280), 215.0)                                     # fundo claro
    cx, cy, a, b = 640, 420, 580, 340                                        # elipse da face
    face = np.zeros(imagem.shape, np.uint8)
    cv2.ellipse(face, (cx, cy), (a, b), 0, 0, 360, 1, cv2.FILLED)
    imagem[face > 0] = 115                                                   # face do disco
    cubo = np.zeros(imagem.shape, np.uint8)
    cv2.ellipse(cubo, (cx, cy), (int(a * 0.55), int(b * 0.55)), 0, 0, 360, 1, cv2.FILLED)
    imagem[cubo > 0] = 175                                                   # cubo claro
    for r in (0.80, 0.85, 0.90, 0.94):                                       # riscos de desgaste
        cv2.ellipse(imagem, (cx, cy), (int(a * r), int(b * r)), 0, 0, 360, 85, 1, cv2.LINE_AA)
    for t in np.deg2rad(np.arange(8) * 45 + 20):                             # trincas radiais
        p1 = (int(cx + a * 0.80 * np.cos(t)), int(cy + b * 0.80 * np.sin(t)))
        p2 = (int(cx + a * 0.93 * np.cos(t)), int(cy + b * 0.93 * np.sin(t)))
        cv2.line(imagem, p1, p2, 40, 2, cv2.LINE_AA)
    luz = np.linspace(30, -30, imagem.shape[1])[None, :]                     # mais luz à esquerda
    imagem = imagem + luz * (face > 0) + gerador.normal(0, 4, imagem.shape)  # e textura leve do metal
    imagem = np.clip(imagem, 0, 255).astype(np.uint8)
    caminho = salvar_imagem(pasta, "disco_sintetico.png", cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR))

    resumo = processar(caminho, pasta / "saida")
    por_quadrante = resumo["trincas_quadrante"]
    passou = all(n == 2 for n in por_quadrante.values())
    print(f"Elipse estimada: {[round(v) for v in resumo['elipse']]} (desenhada: [{cx}, {cy}, {a}, {b}])")
    print(f"Trincas por quadrante: {por_quadrante} (desenhadas: 2 em cada); outras manchas: {resumo['outras_manchas']}")
    print("Teste:", "passou" if passou else "FALHOU")
    return passou


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # acentos no console do Windows
    except (AttributeError, ValueError):
        pass                                       # consoles que não permitem trocar a codificação

    try:                                           # pasta do script (no Spyder e no terminal)
        pasta_script = Path(__file__).resolve().parent
    except NameError:                              # colado num notebook: usa a pasta atual
        pasta_script = Path.cwd()

    parser = argparse.ArgumentParser(description="Detecção de trincas num disco de freio.")
    parser.add_argument("imagem", nargs="?", default=str(pasta_script / "disco_freio_trincado.jpg"),
                        help="foto a analisar (padrão: disco_freio_trincado.jpg na pasta do script)")
    parser.add_argument("--saida", default=str(pasta_script / "imagens"),
                        help="pasta onde as imagens de cada etapa são gravadas (padrão: imagens/)")
    parser.add_argument("--teste", action="store_true", help="valida a sequência num disco sintético")
    args, _ = parser.parse_known_args()            # ignora argumentos extras do Spyder ou do Jupyter

    if args.teste:
        pasta_teste = Path(tempfile.mkdtemp(prefix="teste_trincas_"))   # pasta temporária: não suja o projeto
        print(f"Imagens do teste em: {pasta_teste}")
        if not teste_sintetico(pasta_teste):
            sys.exit(1)        # código 1 no terminal; só sai assim na falha, porque no Spyder
        return                 # um sys.exit aparece como exceção mesmo quando o teste passa
    imprimir_resumo(processar(args.imagem, args.saida))


if __name__ == "__main__":
    main()
