import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import sys
import os
# (pip install python_speech_features==0.6)
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from segmentador import segmentarAudio 
import csv  # para gerar csv

linhas = [] # lista para armazenar os valores finais de cada comando
tabela = []  # lista para armazenar os valores de todos os comandos

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

for comando in range(quantidadeComandos):
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
    for speaker in range(inicioTreino, finalTreino):
        audio = f'{comando_aux}/{speaker}.wav'
        [fs, xi] = wav.read(audio)

        # Muitos microprocessadores causam um estalo no início da gravação, atingindo a máxima amplitude
        estalo = 100
        xi[0:estalo] = 0  # zera as primeiras 100 amostras do audio

        # normalização do amplitude de 16 bits. Isso deixa a amplitude maxima entre um intervalo de 1 e -1
        xi = xi / 32768.0

        x = []
        x = np.append(xi[0], xi[1:] - 0.97 * xi[:-1])  # filtro de pre-enfase

        z = segmentarAudio(x) # recebe o audio segmentado
        
        # tratamento do sinal pos segmentacao        
        z_norm = z / np.max(np.abs(z)) # normalizacao de amostra do áudio segmentado, entre -1 e 1
        zMax = np.max(np.abs(z_norm))
        #print(f"zMax: {zMax}")
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
        #print(f"mfcc_feat tipo: {type(mfcc_feat)}")

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
                linhas.append(mfcc_norm[i][j])
                #linhas.append(';')
                sys.stdout.write(str(mfcc_norm[i][j]))
                sys.stdout.write(';')

        if (comando == 0):
            sys.stdout.write('1;0;0;0;0;0;0;0;0;0;')
            linhas.append('1;0;0;0;0;0;0;0;0;0;')
        if (comando == 1):
            sys.stdout.write('0;1;0;0;0;0;0;0;0;0;')
            linhas.append('0;1;0;0;0;0;0;0;0;0;')
        if (comando == 2):
            sys.stdout.write('0;0;1;0;0;0;0;0;0;0;')
            linhas.append('0;0;1;0;0;0;0;0;0;0;')
        if (comando == 3):
            sys.stdout.write('0;0;0;1;0;0;0;0;0;0;')
            linhas.append('0;0;0;1;0;0;0;0;0;0;')
        if (comando == 4):
            sys.stdout.write('0;0;0;0;1;0;0;0;0;0;')
            linhas.append('0;0;0;0;1;0;0;0;0;0;')
        if (comando == 5):
            sys.stdout.write('0;0;0;0;0;1;0;0;0;0;')
            linhas.append('0;0;0;0;0;1;0;0;0;0;')
        if (comando == 6):
            sys.stdout.write('0;0;0;0;0;0;1;0;0;0;')
            linhas.append('0;0;0;0;0;0;1;0;0;0;')
        if (comando == 7):
            sys.stdout.write('0;0;0;0;0;0;0;1;0;0;')
            linhas.append('0;0;0;0;0;0;0;1;0;0;')
        if (comando == 8):
            sys.stdout.write('0;0;0;0;0;0;0;0;1;0;')
            linhas.append('0;0;0;0;0;0;0;0;1;0;')
        if (comando == 9):
            sys.stdout.write('0;0;0;0;0;0;0;0;0;1;')
            linhas.append('0;0;0;0;0;0;0;0;0;1;')

        #print('')  # escreve as pŕoximas saidas na linha seguinte
        linhas.append('')
        tabela.append(linhas[:])
        linhas.clear()
        #print("tebela")
        #print(tabela[0])
    
# Escrevendo os dados do array para o arquivo CSV 
with open("treinocsvteste", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(tabela)

'''O comando "comm > nome_do_arquivo.csv" deverá ser usado no terminal,
afim de gravar as saidas da função "sys.stdout.write()" em um arqui .csv
exemplo: python3 mfcc.py comm > terino.csv
'''
