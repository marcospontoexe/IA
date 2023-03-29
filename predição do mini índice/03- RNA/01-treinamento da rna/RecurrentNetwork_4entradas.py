import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pybrain   # pip install https://github.com/pybrain/pybrain/archive/0.3.3.zip

import matplotlib.pyplot as plt


from pybrain.tools.shortcuts import buildNetwork	# usado para construir a estrutura de uma rna
from pybrain.structure import RecurrentNetwork, IdentityConnection
from pybrain.datasets import SupervisedDataSet		# método de aprendizagem (supervisionado)
from pybrain.structure import FullConnection		# para fazer a conexão entre as camadas
from pybrain.structure.connections.connection import Connection					# para criar conexões parciais
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

nInputs = 4
hidden_layers = 4
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
#-----criando as camadas com seus neurônios------------------------
input_layer = LinearLayer(nInputs)  	# cria uma camada de entrada
hidden0 = TanhLayer(hidden_layers)			# cria uma camada oculta
hidden1 = TanhLayer(4)			# cria uma camada oculta
output_layer = SoftmaxLayer(nOutputs)  #cria camada de saida
biasHidden0 = BiasUnit()			# cria um bias para a camada oculta 0
biasHidden1 = BiasUnit()			# cria um bias para a camada oculta
biasOut = BiasUnit()			# cria um bias para a camada de saida

#------------adicionando as camadas na rna------------
rede.addInputModule(input_layer)		# adiciona uma camada de entrada
rede.addModule(hidden0)				# adiciona camada oculta
rede.addModule(hidden1)				# adiciona camada oculta
rede.addOutputModule(output_layer)	# adiciona camada de saida
rede.addModule(biasHidden0)				# adiciona um bias a camada oulta
rede.addModule(biasHidden1)				# adiciona um bias a camada oulta
rede.addModule(biasOut)				# adiciona um bias a camada de saida
#---------------------------------------------------------------

#-----------------configurando as conexão entre as camadas------------------
in_to_hidden0 = FullConnection(input_layer, hidden0)		# cria um modo de conexão entre camada de entrada e oculta0
hidden0_to_hidden1 = FullConnection(hidden0, hidden1)		# cria um modo de conexão entre as camadas ocultas
hidden0_to_output = FullConnection(hidden0, output_layer)		# cria um modo de conexão conexão entre a camada oculta0 e camada de saida
biasHidden0_to_hidden0 = FullConnection(biasHidden0, hidden0) # cria um modo de conexão entre bias0 e camada oculta 0
biasHidden1_to_hidden1 = FullConnection(biasHidden1, hidden1) # cria um modo de conexão entre bias1 e camada oculta 1
biasOut_to_out = FullConnection(biasOut, output_layer) # cria um modo de conexão entre biasout e camada de saida
hidden0_to_hidden0 = IdentityConnection(hidden0, hidden0)		# cria um modo de conexão entre as camadas ocultas 0 sem alteração do peso sináptico
hidden1_to_hidden1 = IdentityConnection(hidden1, hidden1)		# cria um modo de conexão entre as camadas ocultas 1 sem alteração do peso sináptico

rede.addConnection(in_to_hidden0)
rede.addConnection(hidden0_to_hidden1)		# adidiona conexão entre as camadas ocultas
rede.addConnection(hidden0_to_output)		# adidiona conexão entre a oculta e a saida
rede.addConnection(biasHidden0_to_hidden0)	# adidiona conexão entre a bias e a camada oculta
rede.addConnection(biasHidden1_to_hidden1)	#adidiona conexão entre a bias e a camada oculta
rede.addConnection(biasOut_to_out)	#adidiona conexão entre a bias e a camada de saida
rede.addRecurrentConnection(hidden0_to_hidden0)	# adicionando as conexões recorrentes
rede.addRecurrentConnection(hidden1_to_hidden1)	# adicionando as conexões recorrentes
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
	base.addSample(X_train[i], y_train[i])


# treinamento da rede neural pelo metodo back propagation
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.0005, momentum = 0.8, batchlearning=False)
#treinamento.trainUntilConvergence(maxEpochs=250, verbose=None, continueEpochs=30, validationProportion=0.25)
epocas = 500
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