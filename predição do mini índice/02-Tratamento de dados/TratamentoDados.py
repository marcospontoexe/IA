import pandas as pd

selecionarCol = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] # seleciona quais colunas o dataframe deve possuir
dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)

dfTratado = pd.DataFrame({"Tamanho", "Cor", "Variação de preço"})
print(dfTratado)
print(type(dfTratado))