import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import os

def segmentador(audio):
    """
    Essa função segmenta um aúdio, separando a região de fala da região de silenciosa.
    :param audio: Áudio a ser tratado.
    :return: Áudio sem a região silênciosa de duração de 1,1 segundos
    """

    xi = audio
    # Muitos microprocessadores causam um estalo no início da gravação, atingindo a máxima amplitude
    estalo = 100
    xi[0:estalo] = 0  # zera as primeiras 100 amostras do audio

    # normalização do amplitude de 16 bits. Isso deixa a amplitude entre um intervalo de 1 e -1
    xi = xi / 32768.0

    x = []
    x = np.append(xi[0], xi[1:] - 0.97 * xi[:-1])  # filtro de pre-enfase

    # n = np.arange(0, len(x))

    # print('speaker: ' + str(speaker))

    win = 3200
    step = 800
    # Calcula a energia contida no vetor x
    # define o tamanho do vetor energia
    energy = np.zeros(int((len(x)-win)/step))
    for i in range(len(energy)):
        tmp = 0
        for j in range(i*step, i*step+win):  # calcula a energia dentro da janela definida
            # somatório das amostras ao quadrado, dentro de cada janela
            tmp = tmp + x[j]*x[j]
        energy[i] = tmp  # vetor com a energia de cada janela

    # indice  onde foi encontrado o maior valor da amostra energy
    iMax = np.argmax(energy)
    vMax = energy[iMax]  # o maior valor da amostra energy
    iMin = np.argmin(energy)  # índice com o menor valor de energy
    # Menor valor de energia, usado para achar o limiar entre região de fala e de  silencio
    vMin = energy[iMin]
    # print(f"iMin; {iMin}")

    # calcula o limiar inferior de energia
    A = 0.03
    B = 0.09
    lim_inferior = A*vMax

    '''
    print('indice da energia maxima: ' + str(iMax))
    print('ENERGIA MÁXIMA: ' + str(vMax))
    print('indice da energia minima: ' + str(iMin))
    print('ENERGIA minima: ' + str(vMin))
    print('limiar inferior de energia: '+str(lim_inferior))
    '''

    if vMin > lim_inferior:  # caso tenha muito ruido de fundo
        lim_inferior = vMin + ((vMax - vMin) * B)
        # print(f"lim_inferior_2 {lim_inferior}")

    # a variável silencio verefica se a veriável energy é menor que o lim_inferior
    silencio = 0
    # para garantir que os vales contidos na regiao falada nao sejam considerados como regiao de silêncio,
    # foi criado a variável amostras_consecutivas
    amostras_consecutivas = 10
    # verificação do limiar inferior de energia para achar start
    for i in range(iMax, 0, -1):  # varre o sinal do pico de energia até o inicio do sinal
        if (energy[i] >= lim_inferior):
            if (i == 1):
                start = 0
        else:  # momento em que a energia do quadro analisado é menor que o limiar inferior de energia
            if (silencio == 0):
                start = i  # marca a amosta onde energy é menor que lim_inferior
            silencio += 1
            # quando ha 10 quadros do áudio pertecentes a região de silencio
            if (silencio == amostras_consecutivas):
                break
    if (iMax == 0):
        start = 1

    if (start == 0):
        start = 0
    else:
        start = (start*step)+win

    stop = start+17600
    if (stop > 32000):  # O tamanho máximo do áudio é de 32000 amostras
        stop = 32000

    # vetor com a região de silencio zerada
    y = np.zeros(len(x))
    for i in range(start, stop):
        y[i] = x[i]

    # vetor com o audio segmentado, comtém apenas a região falada
    # z = np.zeros(stop-start)
    z = np.zeros(17600)
    # for i in range(0,len(z)):
    for i in range(0, (stop-start)):
        z[i] = y[i+start]
    # print(f"len(z): {len(z)}")

    '''
    som = f'aplay {audio}'
    os.system(som)
    '''

    '''
    print('start: '+str(start))
    print('stop: '+str(stop))
    print('\n')
    '''

    # imprime os gŕaficos do vetor de energia, além do audio original, áudio apenas com a regiaão falada, e finalmente o audio segmentado
    plt.subplot(4, 1, 1)
    plt.plot(energy)
    plt.subplot(4, 1, 2)
    plt.plot(x)
    plt.subplot(4, 1, 3)
    plt.plot(y)
    plt.subplot(4, 1, 4)
    plt.plot(z)
    plt.show()

    return z


diretorio_atual = os.path.dirname(os.path.realpath(__file__))  # Diretório do script atual
# Diretório pai do diretório atual
diretorio_pai = os.path.dirname(diretorio_atual)
# Diretório avó do diretório pai
diretorio_avo = os.path.dirname(diretorio_pai)
pasta = f"{diretorio_avo}/Banco_de_palavras"

# pasta = '../../Banco_de_palavras'

for comando in range(10):
    if (comando == 0):
        comando_aux = f'{pasta}/10-Jarbas'
    if (comando == 1):
        comando_aux = f"{pasta}/11-Ligue"
    if (comando == 2):
        comando_aux = f"{pasta}/12-Desligue"
    if (comando == 3):
        comando_aux = f"{pasta}/13-Música"
    if (comando == 4):
        comando_aux = f"{pasta}/14-Sala"
    if (comando == 5):
        comando_aux = f"{pasta}/15-Quarto"
    if (comando == 6):
        comando_aux = f"{pasta}/16-Cozinha"
    if (comando == 7):
        comando_aux = f"{pasta}/17-Ar-Condicionado"
    if (comando == 8):
        comando_aux = f"{pasta}/18-TV"
    if (comando == 9):
        comando_aux = f"{pasta}/19-Café"

    # número de amostras de áudio contida em cada pasta de comando (Ligue, Desligue, Jarbas...)
    for speaker in range(1, 10):
        audio = f'{comando_aux}/{speaker}.wav'
        [fs, xi] = wav.read(audio)

        # Muitos microprocessadores causam um estalo no início da gravação, atingindo a máxima amplitude
        estalo = 100
        xi[0:estalo] = 0  # zera as primeiras 100 amostras do audio
        
        # normalização do amplitude de 15 bits. Isso deixa a amplitude entre um intervalo de 1 e -1
        xi = xi / 32768.0
        

        x = []
        x = np.append(xi[0], xi[1:] - 0.97 * xi[:-1])  # filtro de pre-enfase
        seg = segmentador(x)

        plt.subplot(4, 1, 1)
        plt.plot(seg)
        plt.show()