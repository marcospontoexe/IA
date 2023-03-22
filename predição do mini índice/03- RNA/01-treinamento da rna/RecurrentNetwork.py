import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pybrain   # pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip

import matplotlib.pyplot as plt

from pybrain.tools.shortcuts import buildNetwork	# usado para construir a estrutura de uma rna
from pybrain.structure import RecurrentNetwork
from pybrain.datasets import SupervisedDataSet		# método de aprendizagem (supervisionado)
from pybrain.structure import FullConnection		# para fazer a conexão entre as camadas
from pybrain.supervised.trainers import BackpropTrainer	#método de treiamento supervisionado
#Sigmoid activation functions are used when the output of the neural network is continuous. Softmax activation functions are used when the output of the neural network is categorical.
#https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
from pybrain.structure import SoftmaxLayer, LSTMLayer, SigmoidLayer, TanhLayer, LinearLayer		# funções de ativação
from pybrain.structure import BiasUnit		# bias
from pybrain.tools.customxml import NetworkWriter	# para salvar em xml

seed = 176
np.random.seed(0)
np.random.permutation(seed)
#np.random.seed(seed)

nInputs = 32
hidden_layers = 9
nOutputs = 3

#importacao dos dados para aprendizado
df = pd.read_csv('dfRna.csv', header=None)
tamanhoTreino = int((df.shape[0])*0.75)		# define um tamanho de 75% do banco de dados para usar como treinamento. (25% como daddos para validação)

dfTreino = df.loc[1:tamanhoTreino].copy()		# dataframe para treinar a rna
dfValidacao	= df.loc[(tamanhoTreino+1):].copy()	# dataframe para validação da rna
X_train = dfTreino.iloc[:, 0:nInputs].values    # array com valores de entrada da rna
y_train = dfTreino.iloc[:, nInputs:(nInputs+nOutputs)].values   # array com valores para as saidas da rna

#dfTreino.to_excel("dfTreino.xlsx")      # salva como csv, sem os índices      ###########################################
#dfValidacao.to_excel("dfValidacao.xlsx")      # salva como csv, sem os índices      ###########################################
#print(f"tamanho: {tamanhoTreino}")
#print(X_train)
#print(y_train)

# Construcao da rede neural
#rede = buildNetwork(nInputs, hidden_layers, nOutputs, bias=True, hiddenclass=TanhLayer ou LSTMLayer, outclass=SoftmaxLayer)
rede = RecurrentNetwork()			# cria uma rna recorrente
	#(nInputs,16, 10, 8,5, nOutputs, bias=True, outclass=SoftmaxLayer, outputbias=True)
#-----adicionando camadas--------------------------------------
rede.addInputModule(LinearLayer(nInputs, name='in'))		# adiciona uma camada de entrada
rede.addModule(SigmoidLayer(hidden_layers, name='hidden0'))				# adiciona camada oculta
rede.addModule(SigmoidLayer(8, name='hidden1'))				# adiciona camada oculta
rede.addOutputModule(SoftmaxLayer(nOutputs, name='out'))			#adiciona camada de saida
rede.addModule(BiasUnit(name='biasHidden0'))				# adiciona um bias a camada oulta
rede.addModule(BiasUnit(name='biasHidden1'))				# adiciona um bias a camada oulta
rede.addModule(BiasUnit(name='biasOut'))				# adiciona um bias a camada de saida
#---------------------------------------------------------------

#-----------------configurando as conexão entre as camadas------------------
rede.addConnection(FullConnection(rede['in'], rede['hidden0']))			# conexão entre a camada de entrada e camada oculta
rede.addConnection(FullConnection(rede['hidden0'], rede['hidden1']))		# conexão entre as camadas ocultas
rede.addConnection(FullConnection(rede['hidden1'], rede['out']))		# conexão entre a camada oculta e camada de saida
rede.addConnection(FullConnection(rede['biasHidden0'], rede['hidden0']))	# conexão entre a bias e a camada oculta
rede.addConnection(FullConnection(rede['biasHidden1'], rede['hidden1']))	# conexão entre a bias e a camada oculta
rede.addConnection(FullConnection(rede['biasOut'], rede['out']))	# conexão entre a bias e a camada de saida
rede.addRecurrentConnection(FullConnection(rede['hidden0'], rede['hidden0']))	# adicionando as conexões recorrentes
rede.addRecurrentConnection(FullConnection(rede['hidden1'], rede['hidden1']))	# adicionando as conexões recorrentes
#---------------------------------------------------------------
rede.sortModules()				# inicialização da rna
print(f"RNA: \n{rede}")		# mostra a configuranção da RNA
#print(f"pesos sinápticos da RNA: \n{rede.params}")		# mostra o valor dos pesos sinápticos da RNA
#print(f"camada de entrada: {rede['in']}")		# mostra o módo da rede de entrada
#print(f"camada de oculta0: {rede['hidden0']}")
#print(f"camada de saida: {rede['out']}")
#print(f"biasHidden0: {rede['biasHidden0']}")
#print(f"biasOut: {rede['biasOut']}")
#print(f"recurrent: {rede['recurrent']}")			# para escolher entre RecurrentNetwork ou FeedForwardNetwork

base = SupervisedDataSet(nInputs, nOutputs)

# insere os dados na rede neural
for i in range(len(X_train)):
	base.addSample(X_train[i],y_train[i])


# treinamento da rede neural pelo metodo back propagation
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.1, momentum = 0.005, batchlearning=False)
#treinamento.trainUntilConvergence(maxEpochs=250, verbose=None, continueEpochs=30, validationProportion=0.25)
epocas = 2000
learning_rate = np.zeros(epocas)
for i in range(1, epocas):
    erro = treinamento.train()		# treina a rna pelo método de épocas (usar 'trainer.trainUntilConvergence()' para o método de convergência)
    learning_rate[i-1] = erro		# learning_rate é usado apenas para plotar o gráfico
    print(f"Erro {i}: {erro}")


# imprime a matriz confusao de treinamento
print('matriz confusao de treino: ')

matrizConfusao = np.zeros((3,3))		# cria um array de duas dimensões 3x3
for i in range(len(X_train)):
	y_certo = np.argmax(y_train[i])				# retorna o índice (0, 1, 2, 3...)
	y_predito = np.argmax(rede.activate(X_train[i]))		# retorna o índice (0, 1, 2, 3...)
	#print(y_certo)
	#print(y_predito)
	#print('---')
	matrizConfusao[y_certo][y_predito] += 1
print(matrizConfusao)



# imprime a matriz confusao de teste
X_train2 = dfValidacao.iloc[:, 0:nInputs].values
y_train2 = dfValidacao.iloc[:, nInputs:(nInputs+nOutputs)].values
print('matriz confusao de validação :')
matrizConfusao2 = np.zeros((3,3))
for i in range(len(X_train2)):
	y_certo2 = np.argmax(y_train2[i])
	y_predito2 = np.argmax(rede.activate(X_train2[i]))
	#print(y_certo)
	#print(y_predito)
	#print('---')
	matrizConfusao2[y_certo2][y_predito2] += 1
print(matrizConfusao2)

#mostra a curva da taxa de aprendizagem
plt.plot(learning_rate)
plt.show()

# gera um arquivo XML
NetworkWriter.writeToFile(rede, 'model.xml')


'''
class pybrain.tools.neuralnets.saveNetwork('teste.csv')
NetworkWriter.writeToFile(rede, 'model.xml')
#https://stackoverflow.com/questions/12050460/neural-network-training-with-pybrain-wont-converge
'''