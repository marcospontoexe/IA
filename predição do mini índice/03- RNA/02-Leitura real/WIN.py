import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import pybrain   # pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip

import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork	# usado para construir a estrutura de uma rna
from pybrain.datasets import SupervisedDataSet		# método de aprendizagem (supervisionado)
from pybrain.supervised.trainers import BackpropTrainer	#método de treiamento supervisionado
from pybrain.structure.modules import SoftmaxLayer, LSTMLayer	# funções de ativação
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import BiasUnit		# bias
from pybrain.tools.customxml import NetworkWriter	# para salvar em xml
from pybrain.tools.customxml import NetworkReader	## para ler em xml

from random import randrange, uniform,randint
import matplotlib.cm as cm
import sys
import os
#import math
#import matplotlib.ticker as mticker
#from functools import partial
#from subprocess import call

seed = 176
np.random.seed(0)
np.random.permutation(seed)
#np.random.seed(seed)

nInputs = 36
hidden_layers = 9
nOutputs = 3
rede = NetworkReader.readFrom('model.xml')		# importa o arquivo com os pesos sinápticos


while(True):


	# quando for 17h ler ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] e gerar dataframe 'dfBruto'
    dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)
    filtro = dfBruto["<TIME>"] != "18:00:00"  # filtra apenas as velas das 9h às 17h
    dfTratado = dfBruto[filtro].loc[:,["<DATE>", "<TIME>"]].copy()  # copia para novo dataframe apenas as colunas "<DATE>" e "<TIME>", mantendo os mesos índices

    # -----verificando se o dataframe contem todos as velas 9h-17h) do dia-------------
    condicao = dfTratado['<DATE>'].value_counts() < 9  # TRUE para dias que tem menos de 8 velas (das 9h às 17h)
    datas = ((dfTratado['<DATE>'].value_counts()[condicao]).index).values  # datas dos dias incompletos
    print('As seguintes datas estão incompletas e foram apagadas:')
    for i in range(0, len(datas)):  # apagando o dia incompleto
        print(f"{datas[i]}")  # mostra a data excluida do dftratado
        datasIndice = ((dfTratado['<DATE>']) == datas[
            i])  # recebe um dataframe com 'True' nas linhas do dia comparado com a data
        lista = dfTratado[datasIndice]  # recebe um dataframe com as linhas do dia comparado
        listaIndice = lista.index  # recebe uma lista com os índices do dataframe 'lista'
        dfTratado.drop(listaIndice, inplace=True)  # apaga as linhas dos dias incompletos
    # ---------------------------------------------------------

    # -----Criando novas colunas-------------------------
    dfTratado['<TAMANHO>'] = np.nan  # tam (tamanho da vela)
    dfTratado['<TAMANHO NORMALIZADO>'] = np.nan  # tamNor (normalização do tamanho da vela)
    dfTratado['<VARIAÇÃO DO PREÇO>'] = np.nan  # var (variação de preço)
    dfTratado['<VARIAÇÃO DO PREÇO NORMALIZADO>'] = np.nan  # varNor (normalização da variação de preço)
    dfTratado['<PREÇO>'] = np.nan  # preco (valor de abertura normalizado)
    dfTratado['<VOLUME NORMALIZADO>'] = np.nan  # volNor (volume normalizado)
    # -----------------------------------------------

    # ----------------Tratando o dataframe------------------
    dfTratado['<TAMANHO>'] = dfBruto['<CLOSE>'] - dfBruto['<OPEN>']  # tam (tamanho da vela)
    dfTratado['<VARIAÇÃO DO PREÇO>'] = dfBruto['<HIGH>'] - dfBruto['<LOW>']  # var (variação de preço)
    somaTamanho = 0
    for indice, coluna in dfTratado.iterrows():
        if (coluna["<TIME>"] == "09:00:00"):  # para as velas das 9h às 16h
            varMax = float(dfBruto.loc[indice:indice + 7,["<HIGH>"]].max().values)  # maior preço de negociação no dia (das 9h às 16h)
            ivol = float(dfBruto.loc[indice:indice+7,["<VOL>"]].max().values)  # (maior volume encontrado no dia, das 9h às 16h)
            ivar = int(dfTratado.loc[indice:indice+7,["<VARIAÇÃO DO PREÇO>"]].max().values)  # maior variação de preço do dia das 9h às 16h

            dfTratado.loc[indice:indice + 8, ["<TAMANHO NORMALIZADO>"]] = ((dfTratado.loc[indice:indice + 8,["<TAMANHO>"]]).values) / ivar  # tamNor (normalização do tamanho da vela)
            dfTratado.loc[indice:indice + 8, ["<VARIAÇÃO DO PREÇO NORMALIZADO>"]] = (((dfTratado.loc[indice:indice + 8,["<VARIAÇÃO DO PREÇO>"]]).values)) / ivar  # varNor (normalização da variação de preço)
            dfTratado.loc[indice:indice + 8, ['<PREÇO>']] = (((dfBruto.loc[indice:indice + 8, ["<OPEN>"]]).values)) / varMax  # preco (valor de abertura normalizado)
            dfTratado.loc[indice:indice + 8, ['<VOLUME NORMALIZADO>']] = (((dfBruto.loc[indice:indice + 8, ["<VOL>"]]).values)) / ivol  # volNor (volume normalizado)
    # ----------------------------------------------------

    # -----Novo dataframe apenas com as linha últeis---------------------
    colunasDf = ["<PREÇO>", "<TAMANHO NORMALIZADO>", "<VARIAÇÃO DO PREÇO NORMALIZADO>", "<VOLUME NORMALIZADO>"]  # seleciona quais colunas o dataframe deve possuir
    dfLimpo = dfTratado[colunasDf].copy()
    # dfLimpo.to_excel("dfLimpo.xlsx")      # salva como csv, sem os índices
    # --------------------------------------------------------------------

    # -----Gerando o dataframe para a RNA---------
    dfRna = pd.DataFrame({  # cria um dataframe com apenas uma linha
        'X1': [1], 'X2': [1], 'X3': [1], 'X4': [1], 'X5': [1], 'X6': [1], 'X7': [1], 'X8': [1], 'X9': [1], 'X10': [1],
        'X11': [1], 'X12': [1], 'X13': [1], 'X14': [1], 'X15': [1], 'X16': [1], 'X17': [1], 'X18': [1], 'X19': [1],
        'X20': [1], 'X21': [1], 'X22': [1], 'X23': [1], 'X24': [1], 'X25': [1], 'X26': [1], 'X27': [1], 'X28': [1], 'X29': [1],
        'X30': [1], 'X31': [1], 'X32': [1], 'X33': [1], 'X34': [1], 'X35': [1], 'X36': [1]
    })
    contColuna = 0  # iteração para incluir colunas no dataframe 'dfRNA'
    for linha, dado in dfLimpo.iterrows():  # transpondo a matriz
        listaLinha = (dado[:].values)  # recebe uma lista em float com os valores da linha atual do dataframe
        for i in range(0, 4):   # a cada quatro colunas
            contColuna = contColuna + 1
            dfRna.loc[0, [f"X{contColuna}"]] = listaLinha[i]

    #--------Adicionando os valorea à entrada da rna----------
    X_train = dfRna.values  # array com valores de entrada da rna
    saida = rede.activate(X_train)      # lista com o valor de cada saida
    actv = np.argmax(saida)             # índice onde ocorreu o maior valor
    if(actv == 0):      # COMPRA
        print('COMPRAR!')
    elif (actv == 1):  # LATERAL
        print('LATERAL!')
    if (actv == 2):  # VENDA
        print('VENDER!')
    else:
        print("ERRO!")
    #---------------------------------------------------------





