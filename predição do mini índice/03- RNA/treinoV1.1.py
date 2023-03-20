import pandas as pd
import numpy as np
import pybrain    # pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip
#from pybrain.structure import FeedForwardNetwork
#https://stackoverflow.com/questions/46744105/running-neural-network-pybrain
import matplotlib.pyplot as plt


nInputs = 36
hidden_layers = 9
nOutputs = 3

#importacao dos dados para aprendizado
df = pd.read_csv('dfRna.csv', header=None)
tamanhoTreino = int((df.shape[0])*0.8)		# define um tamanho de 80% do banco de dados para usar como treino (20% como validação)
dfTreino = df.loc[1:tamanhoTreino].copy()		# dataframe para treinar a rna
dfValidacao	= df.loc[(tamanhoTreino+1):].copy()	# dataframe para validação da rna
X_train = dfTreino.iloc[:, 0:nInputs].values    # array com valores de entrada da rna
y_train = dfTreino.iloc[:, nInputs:(nInputs+nOutputs)].values   # array com valores para as saidas da rna


#dfTreino.to_excel("dfTreino.xlsx")      # salva como csv, sem os índices
#dfValidacao.to_excel("dfValidacao.xlsx")      # salva como csv, sem os índices





