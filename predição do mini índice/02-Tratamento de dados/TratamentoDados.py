import pandas as pd
import numpy as np

selecionarCol = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] # seleciona quais colunas o dataframe deve possuir
dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)
filtro = dfBruto["<TIME>"] != "18:00:00"        # filtra apenas as velas das 9h às 17h
dfTratado = dfBruto[filtro].loc[:, ["<DATE>", "<TIME>"]].copy()     # copia para novo dataframe apenas as colunas "<DATE>" e "<TIME>", mantendo os mesos índices
dfTratado = dfBruto[filtro].loc[:, ["<DATE>", "<TIME>"]].copy()  # copia para novo dataframe apenas as colunas "<DATE>" e "<TIME>", mantendo os mesos índices

#-----verificando se o dataframe contem todos os horários dos dia (9h-17h)-------------
flag = 0
for indice, coluna in dfTratado.iterrows():
    if(flag == 0):
        temp = coluna['<DATE>']
        hora = 9
        flag = 1

    try:
        if( (coluna['<DATE>'] == temp) and (coluna['<TIME>'] == (f"{hora:>02}:00:00")) ):
            if (hora == 17):
                flag = 0
    except Exception as erro:  # mostra qual foi o erro retornado pela exceção
        print(f"A classe do erro encontrado foi {erro.__class__}!")
        print(f"O erro encontrado foi {erro.__cause__}!")

    else:
        print(f"ERRO na data {coluna['<DATE>']}, às {coluna['<TIME>']}")
        exit()      # para o programa


    hora = hora + 1




#---------------------------------------------------------


#-----Criando novas colunas-------------------------
dfTratado['<TAMANHO>'] = np.nan
dfTratado['<TAMANHO NORMALIZADO>'] = np.nan
dfTratado['<VARIAÇÃO DO PREÇO>'] = np.nan
dfTratado['<PREÇO>'] = np.nan
dfTratado['<VOLUME>'] = np.nan
#-----------------------------------------------


#----------------Tratando o dataframe------------------
for indice, coluna in dfTratado.iterrows():
    if (coluna["<TIME>"] == "09:00:00"):    # para as velas das 9h às 16h
        varMax = int(dfBruto.loc[indice:indice+7, ["<HIGH>"]].max().values)    # maior preço de negociação do dia (das 9h às 16h)



    open = int((dfBruto.loc[[indice],["<OPEN>"]]).values)       # recebe o valor em 'int'
    close = int(((dfBruto.loc[[indice],["<CLOSE>"]])).values)   # recebe o valor em 'int'
    high = int(((dfBruto.loc[[indice],["<HIGH>"]])).values)   # recebe o valor em 'int'
    low = int(((dfBruto.loc[[indice],["<LOW>"]])).values)   # recebe o valor em "int"


    dfTratado.loc[[indice], ['<TAMANHO>']] = close - open       # tam (tamanho da vela)
    dfTratado.loc[[indice], ['<VARIAÇÃO DO PREÇO>']] = high - low  # var (variação de preço)
    dfTratado.loc[[indice], ['<PREÇO>']] = open/varMax  # preco (valor de abertura normalizado)

    if (coluna["<TIME>"] == "17:00:00"):    # para as velas das 9h às 17h
        ivar = int(dfTratado.loc[indice-8:indice-1, ["<VARIAÇÃO DO PREÇO>"]].max().values)    # maior variação de preço do dia das 9h às 16h
        #ivar = int(dfTratado.loc[indice-8:indice, ["<VARIAÇÃO DO PREÇO>"]].max().values)    # maior variação de preço do dia das 9h às 17h

        for i in range(indice-8, indice+1):        # das 9h ate 17h
            #dfBruto.loc[[i],["<TAMANHO NORMALIZADO>"]] = (int(dfTratado.loc[[i], ['<TAMANHO>']].values)) / ivar   # tamNor (normalização do tamanho da vela)
            print(dfTratado.loc[[i], ['<TIME>', "<DATE>"]])
    #dfTratado.loc[[indice], ['<VOLUME>']] = 4  # vol (volume de ordens)
#----------------------------------------------------

#print(dfTratado.head(30))


'''
#-----Gerando o dataframe para a RNA---------
linha = 0
for indice, coluna in dfBruto.iterrows():
    if (coluna["<TIME>"] == "09:00:00"):    # faz tratamento de dados apenas nas velas das 9h até 16h
        for i in range (0, 8):
            dfTratado.loc[[linha], ['<MAIOR VARIAÇÃO DO PREÇO>']] = 1       # ivar (maior variação de preço do dia)
            dfTratado.loc[[linha], ['<TAMANHO>']] = 2                       # tam (tamanho da vela)
            dfTratado.loc[[linha], ['<TAMANHO_NORMALIZADO>']] = 3           # tamNor (normalização do tamanho da vela)
            dfTratado.loc[[linha], ['<VARIAÇÃO DO PREÇO>']] = 4             # var (variação de preço)
            dfTratado.loc[[linha], ['<VARIAÇÃO DO PREÇO NORMALIZADO>']] = 5 #varNor (normalização da variação de preço)
            dfTratado.loc[[linha], ['<MAIOR PREÇO>']] = 6                   # varMax (maior preço de negociação do dia)
            dfTratado.loc[[linha], ['<PREÇO>']] = 7                         # preco (valor de abertura normalizado)
            dfTratado.loc[[linha], ['<VOLUME>']] = 8                        # vol (volume de ordens)
            dfTratado.loc[[linha], ['<MAIOR VOLUME>']] = 9                  # ivol (maior volume encontrado no dia)
            dfTratado.loc[[linha], ['<VOLUME NORMALIZADO>']] = 10           # volNor (volume normalizado)
            linha = linha+1
print(f"{dfTratado}")
'''


