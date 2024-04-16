import numpy as np
import os

speaker = input("Digite o nome do voluntário: ")
comandos = ["JARBAS", "LIGUE", "DESLIGUE", "MÚSICA", "SALA",
            "QUARTO", "COZINHA", "AR-CONDICIONADO", "TV", "CAFE"]
repeticoes = 3

for i in range(0, len(comandos)):
    for rodada in range(0, repeticoes):  # grava os comandos têrs vezes
        print(f"\nDIGA {comandos[i]}...\n")
        os.system(
            f"arecord -D hw:0,1 -f dat -d 2 -c 2 -r 16000 {speaker}-{comandos[i]}-{rodada}.wav")

'''
-D hw:0,1: Define o dispositivo de áudio de entrada a ser usado. No caso, hw:0,1 refere-se ao dispositivo de áudio específico na posição 0, subdispositivo 1.
-f dat: Especifica o formato de áudio para a gravação. Neste caso, dat indica que os dados de áudio serão gravados em formato de dados crus.
-d 2: Define a duração da gravação em segundos. Neste exemplo, a gravação terá uma duração de 2 segundos.
-c 2: Especifica o número de canais de áudio. Aqui, 2 indica que a gravação será estéreo, ou seja, terá dois canais.
-r 16000: Define a taxa de amostragem de áudio em Hz. Neste caso, a taxa de amostragem é de 16.000 Hz.
{speaker}-{comandos[i]}-{rodada}.wav: Nome do arquivo de saída onde o áudio gravado será salvo. 
'''
