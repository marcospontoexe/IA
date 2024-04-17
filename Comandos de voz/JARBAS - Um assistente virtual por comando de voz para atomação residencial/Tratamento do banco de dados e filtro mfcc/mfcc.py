import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import sys
import os
# (pip install python_speech_features==0.6)
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

quantidadeComandos = 10 # determina a quantidade de comandos, para este projeto existem 10 comandos
inicioTreino = 1        # valor inicial dos comandos usados para treinamento da rna
finalTreino = 2         # valor final dos comandos usados para treinamento da rna
inicioTeste = 3         # valor inicial dos comandos usados para validação da rna
finalTeste = 4          # valor final dos comandos usados para validação da rna

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
    for speaker in range(51, 61):
        audio = f'{comando_aux}/{speaker}.wav'
        [fs, xi] = wav.read(audio)
        # normalização do amplitude de 15 bits. Isso deixa a amplitude entre um intervalo de 1 e -1
        xi = xi / 32768.0
        # Muitos microprocessadores causam um estalo no início da gravação, atingindo a máxima amplitude
        estalo = 100
        xi[0:estalo] = 0  # zera as primeiras 100 amostras do audio

        x = []
        x = np.append(xi[0], xi[1:] - 0.97 * xi[:-1])  # filtro de pre-enfase

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

        # calcula o limiar inferior de energia
        A = 0.03
        B = 0.09
        lim_inferior = A*vMax

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

        # tratamento do sinal pos segmentacao
        # normalizacao de amplitude do áudio segmentado, entre -1 e 1
        z_norm = z / np.max(np.abs(z))
        zMax = np.max(np.abs(z_norm))
        # print(f"z_norm maior: {zMax}")
        # print(f"z_norm len: {len(z_norm)}")
        # print(f"z_norm tipo: {type(z_norm)}")

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # extracao de caracteristicas cepstrais de cada audio
        '''aqui o sinal passa pela função mfcc. O sinal é dividido em chunks, 10 pedacos do mesmo tamanho.
		cada pedaço  é filtrado por 13 filtros da escala MEL, extraindo caracteristicas desse audio, que estão
		relacionadas com o espectro de frequência. Resultando em 130 amostras de um sinal de áudio que 
		continha 32000 amostras.'''

        chunks = 10
        tempo_total = (float)(len(z_norm))/fs
        quantidadeCoeficienteMFCC = 13 # cada coeficiente possui a energia de uma faixa de frequencia

        # mfcc_feat = mfcc(z_norm,fs,winlen=tempo_total/chunks,winstep=tempo_total/#chunks,numcep=13,nfilt=40,nfft=16384,lowfreq=50,preemph=0,appendEnergy=True)
        mfcc_feat = mfcc(z_norm, fs, winlen=tempo_total/chunks, winstep=tempo_total/chunks, numcep=quantidadeCoeficienteMFCC,
                         nfilt=26, nfft=16384, lowfreq=50, preemph=0, appendEnergy=True, winfunc=np.hamming)
        # mfccMax = np.max(np.abs(mfcc_feat))
        # print(f"mfcc_feat max: {mfccMax}")
        # print(f"mfcc_feat : {mfcc_feat}")
        # print(f"mfcc_feat len: {len(mfcc_feat)}")
        # print(f"mfcc_feat tipo: {type(mfcc_feat)}")

        # normaliza os valores cepstrais entre 1 e -1
        mfcc_norm = mfcc_feat / np.max(np.abs(mfcc_feat))
        # mfccMax_norm = np.max(np.abs(mfcc_norm))
        # print(f"mfcc_norm : {mfcc_norm}")
        # print(f"mfcc_norm maior: {mfccMax_norm}")
        # print(f"mfcc_norm len: {len(mfcc_norm)}")
        # print(f"mfcc_norm tipo: {type(mfcc_norm)}")

        # transformando a matriz de 13x10 amostras, em um vetor de uma dimensão apenas, com 130 amostras.
        for i in range(0, chunks):
            for j in range(0, quantidadeCoeficienteMFCC):
                sys.stdout.write(str(mfcc_norm[i][j]))
                sys.stdout.write(';')

        if (comando == 0):
            sys.stdout.write('1;0;0;0;0;0;0;0;0;0;')
        if (comando == 1):
            sys.stdout.write('0;1;0;0;0;0;0;0;0;0;')
        if (comando == 2):
            sys.stdout.write('0;0;1;0;0;0;0;0;0;0;')
        if (comando == 3):
            sys.stdout.write('0;0;0;1;0;0;0;0;0;0;')
        if (comando == 4):
            sys.stdout.write('0;0;0;0;1;0;0;0;0;0;')
        if (comando == 5):
            sys.stdout.write('0;0;0;0;0;1;0;0;0;0;')
        if (comando == 6):
            sys.stdout.write('0;0;0;0;0;0;1;0;0;0;')
        if (comando == 7):
            sys.stdout.write('0;0;0;0;0;0;0;1;0;0;')
        if (comando == 8):
            sys.stdout.write('0;0;0;0;0;0;0;0;1;0;')
        if (comando == 9):
            sys.stdout.write('0;0;0;0;0;0;0;0;0;1;')

        print('')  # escreve as pŕoximas saidas na linha seguinte

'''O comando "comm > nome_do_arquivo.csv" deverá ser usado no terminal,
afim de gravar as saidas da função "sys.stdout.write()" em um arqui .csv
exemplo: python3 mfcc.py comm > terino.csv
'''
