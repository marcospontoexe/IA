import pandas as pd

nInputs = 36
hidden_layers = 9
nOutputs = 3

#importacao dos dados para aprendizado
df = pd.read_csv('dfRna.csv', header=None)
print(type(df))
tamanhoTreino = int((df.shape[0])*0.8)		# define um tamanho de 80% do banco de dados para usar como treino (20% como validação)
dfTreino = df.loc[:tamanhoTreino].copy()		# dataframe para treinar a rna
print(type(dfTreino))
dfValidacao	= df.loc[(tamanhoTreino+1):].copy()	# dataframe para validação da rna
print(type(dfValidacao))
X_train = df.iloc[:, 0:nInputs].values
print(X_train)
y_train = df.iloc[:, nInputs:(nInputs+nOutputs)].values

