import pandas as pd

'''#import pybrain3
import numpy as np

import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, LSTMLayer
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import BiasUnit
from pybrain.tools.customxml import NetworkWriter	#necessário pybrain v0.3.3'''


'''
seed = 176
np.random.seed(0) 
np.random.permutation(seed)
#np.random.seed(seed)
'''
nInputs = 36
hidden_layers = 9
nOutputs = 3
#importacao dos dados para aprendizado
df = pd.read_csv('dfRna.csv', header=None)
tamanhoTreino = int((df.shape[0])*0.8)		# define um tamanho de 80% do banco de dados para usar como treino (20% como validação)
dfTreino = df.loc[:tamanhoTreino].copy		# dataframe para treinar a rna
dfValidacao	= df.loc[(tamanhoTreino+1):].copy	# dataframe para validação da rna
X_train = df.iloc[:, 0:nInputs].values
y_train = df.iloc[:, nInputs:(nInputs+nOutputs)].values



# Construcao da rede neural
#rede = buildNetwork(nInputs, hidden_layers, nOutputs, bias=True, hiddenclass=TanhLayer ou LSTMLayer, outclass=SoftmaxLayer)
rede = buildNetwork(nInputs, hidden_layers, nOutputs, bias=True, outclass=SoftmaxLayer)
base = SupervisedDataSet(nInputs, nOutputs)

# insere os dados na rede neuraloftmax
for i in range(len(X_train)):
	base.addSample(X_train[i],y_train[i])

# treinamento da rede neural pelo metodo back propagation
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.06, momentum = 0.005, batchlearning=False)
#treinamento.trainUntilConvergence(maxEpochs=250, verbose=None, continueEpochs=30, validationProportion=0.25)
epocas = 30

learning_rate = np.zeros(epocas)
for i in range(1, epocas):
    erro = treinamento.train()
    learning_rate[i-1] = erro
   #if i % 50 == 0:
    print("Erro "+str(i)+": %s" % erro)
      
# imprime a matriz confusao de treinamento
print('matriz confusao de treino: ')
matrizConfusao = np.zeros((10,10))
for i in range(len(X_train)):
	y_certo = np.argmax(y_train[i])
	y_predito = np.argmax(rede.activate(X_train[i]))
	#print(y_certo)
	#print(y_predito)
	#print('---')
	matrizConfusao[y_certo][y_predito] += 1
print(matrizConfusao)

# importacao dos dados CSV para o aprendizado
df2 = pd.read_csv('teste.csv', header=None, sep=';')
X_train2 = df2.iloc[:, 0:nInputs].values
y_train2 = df2.iloc[:, nInputs:(nInputs+nOutputs)].values

#normalizacao do csv de teste
#X_teste_norm = X_train2/np.max(np.abs(X_train2))

# imprime a matriz confusao de teste
print('matriz confusao de teste :')
matrizConfusao2 = np.zeros((10,10))
y_certo2 = 0
y_predito2 = 0
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
print(rede)
'''#class pybrain.tools.neuralnets.saveNetwork('teste.csv')
NetworkWriter.writeToFile(rede, 'model.xml')
#https://stackoverflow.com/questions/12050460/neural-network-training-with-pybrain-wont-converge
'''

'''
for i in range(1, 60):
    erro = treinamento.train()
    if i % 50 == 0:
        print("Erro: %s" % erro)
0,01 taxa
36 hidden
'''

