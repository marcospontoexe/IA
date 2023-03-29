import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler         # para normalizar os dados
from sklearn.model_selection import train_test_split    # divi o dataset em conjunto de treino e validação
from sklearn.metrics import r2_score        # para medir a eficiencia do modelo treinado
from sklearn.linear_model import SGDRegressor   # algorítimo de regressão linear usando a Descida do Gradiente Estocastico
from sklearn.neural_network import MLPRegressor     # regressão com perceptron de multiplas camadas

df  = pd.read_csv('auto-mpg.csv')
'''
#-----Plotando um gráfico com 'weight' no eixo X, e 'mpg' no eixo Y--------
plt.scatter(df[['weight']], df[['mpg']])    # seleciona os eixos X e Y
plt.xlabel('peso (libras)')             #título do eixo X
plt.ylabel('autonomia (mpg)')    #título do eixo Y
plt.title('relação entre peso e autonomia de veículos')
plt.show()          #mostra o gráfico
'''
#--------pré processamento------
x = df.loc[:,['weight']]
Y = df.loc[:,['mpg']]

#----transformando para k/l----
x['peso [quilo]'] = x['weight'] * 0.453592         # libras para quilo
x.drop('weight', axis=1, inplace=True)    # apaga a coluna 'weight'
Y['autonomia [k/l]'] = Y['mpg'] * 0.425144               # m/g para k/l
Y.drop('mpg', axis=1, inplace=True)    # apaga a coluna 'weight'


#-----Plotando um gráfico com 'peso [quilo]' no eixo X, e 'autonomia [k/l]' no eixo Y--------
plt.scatter(x[['peso [quilo]']], Y[['autonomia [k/l]']])    # seleciona os eixos X e Y
plt.xlabel('peso (libras)')             #título do eixo X
plt.ylabel('autonomia (mpg)')    #título do eixo Y
plt.title('relação entre peso e autonomia de veículos')
plt.show()          #mostra o gráfico


#----normalizando de padronização dos dados------------------
#A transformação padrão de padronização faz com que os dados tenham média zero e desvio padrão igual a 1
padronizacao = StandardScaler()
padronizacao.fit(x)   # cria uma escala em função do valores de x
xNorm = padronizacao.transform(x)           # recebe um array com os valores normlizados
#padronizacao.fit(y)   # cria uma escala em função do valores de y
#yNorm = padronizacao.transform(y)
y = (Y.loc[:,['autonomia [k/l]']].values).ravel()     # recebe uma array


'''
#----normalizando de reescala dos dados------------------
#faz com que os dados tenham valores entr 0 e 1
escala = MinMaxScaler()
escala.fit(x)   # cria uma escala em função do valores de x
xNorm = escala.fit_transform(x)
 '''

#----- dividindo o dataset em amostras de treinamento e amostras e validação ------
xNormTrain, xNormTest, yTrain, yTeste = train_test_split(xNorm, y, test_size=0.25)  # 25% do dataset é dividido para validação

#----criando uma regressão linear com MLP ------------------------
rna = MLPRegressor(hidden_layer_sizes=(10, 5),      # duas camadas ocultas, com 10 e 5 neurênios em cada
                   max_iter=2000,                   # qunidade de épocas
                   tol=0.0000001,                   # usada para determinar o estado de cenvergencia do erro na saida
                   learning_rate_init=0.1,          # taxa de aprendizagem inicial
                   solver= 'sgd',                   # método de Descida do Gradiente Estocastico
                   activation='logistic',           # fnção da ativação logística (sigmoid)
                   learning_rate='constant',
                   verbose=2)       #mostra oque acontece a cada época de treinamento


rna.fit(xNormTrain, yTrain) # treina a rna com as entrada e saida


#---------criando uma regressão linear-----
regLinear = SGDRegressor(max_iter=2000,         # numero maximo de épocas
                         tol=0.0000001,         # usada para determinar o estado de cenvergencia do erro na saida
                         eta0=0.1,              # taxa de aprendizagem
                         learning_rate='constant',
                         verbose=2)

regLinear.fit(xNormTrain, yTrain) # treina a rna com as entrada e saida

#------previsão da rna e da progressão linear------
YRna = rna.predict(xNormTest)
YRegLinear = regLinear.predict(xNormTest)

#-----calculo do r^2 (quanto mais próximo de 1 melhor)-----------------------
r2Rna = r2_score(yTeste, YRna)
r2Rl = r2_score(yTeste, YRegLinear)
print(f"R2 da RNA:\n{r2Rna}")
print(f"R2 da regressão linear:\n{r2Rl}")

#-------plotando uma comparação entre os métodos-----
xTeste = padronizacao.inverse_transform(xNormTest).ravel()      # desnormalização

plt.scatter(xTeste, yTeste, alpha=0.5, label="Real")    # seleciona os eixos X e Y
plt.scatter(xTeste, YRna, alpha=0.5, label="MLP")    # seleciona os eixos X e Y
plt.scatter(xTeste, YRegLinear, alpha=0.5, label="Regre Linear")    # seleciona os eixos X e Y
plt.xlabel('peso (Kg)')             #título do eixo X
plt.ylabel('autonomia (Km/l)')    #título do eixo Y
plt.title('Comparação dos algorítimos previstos')
plt.legend(loc=1)
plt.show()          #mostra o gráfico


#-------prevendo a saida-----------
previsao = np.array([[1250, 1500, 1600]])           # previsão de autonomia para valores diferentes de kg
prevNormalizado = padronizacao.transform(previsao.T)    #normaliza e transpõe de linha para coluna

rnaPrevisao = rna.predict(prevNormalizado)      # resutado da rna
rlPrevisao = regLinear.predict(prevNormalizado)      # resutado da regressão linear

print(f"previsão da RNA: \n{rnaPrevisao}")
print(f"previsão da regre linear: \n{rlPrevisao}")


#-----plotando a previsão------------------
plt.scatter(x, y, label="Real")    # seleciona os eixos X e Y
plt.scatter(previsao, rnaPrevisao, alpha=0.5, label="MLP")    # seleciona os eixos X e Y
plt.scatter(previsao, rlPrevisao, alpha=0.5, label="Regre Linear")    # seleciona os eixos X e Y
plt.xlabel('peso (Kg)')             #título do eixo X
plt.ylabel('autonomia (Km/l)')    #título do eixo Y
plt.title('Resultado da previsão')
plt.legend(loc=1)
plt.show()          #mostra o gráfico


#--------Buscando os melhores hiperparâmetros-------
hid0        # quantidade de enurônios na camada uculta para 
hid1