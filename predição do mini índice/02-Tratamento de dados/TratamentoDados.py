import pandas as pd

selecionarCol = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] # seleciona quais colunas o dataframe deve possuir
dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)

dfTratado = dfBruto.loc[:, ["<DATE>", "<TIME>"]]    # recebe apenas as colunas "<DATE>" e "<TIME>"

print(dfTratado)
print('--------------------------------')
#-----Inserindo colunas do dfTratado---------
linha = 0
for indice, coluna in dfBruto.iterrows():
    if (coluna["<TIME>"] == "09:00:00"):
        for i in range (0, 8, 1):
            dfTratado.loc[[linha], ['<TAMANHO>']] = 1
            dfTratado.loc[[linha], ['<COR>']] = 1
            dfTratado.loc[[linha], ['<VARIAÇÃO DO PREÇO>']] = 1
            dfTratado.loc[[linha], ['<MAIOR VARIAÇÃO DO PREÇO>']] = 1
            dfTratado.loc[[linha], ['<VARIAÇÃO DO PREÇO NORMALIZADO>']] = 1
            dfTratado.loc[[linha], ['<MAIOR PREÇO>']] = 1
            dfTratado.loc[[linha], ['<MAIOR PREÇO>']] = 1
            dfTratado.loc[[linha], ['<TAMANHO>']] = 1
            dfTratado.loc[[linha], ['<TAMANHO>']] = 1
            dfTratado.loc[[linha], ['<TAMANHO>']] = 1
            linha = linha+1
print(f"{dfTratado['<TAMANHO>']}")
        #dfTratado.loc[[indice],['<TAMANHO>']] =


