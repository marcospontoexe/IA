import pandas as pd
import numpy as np

selecionarCol = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] # seleciona quais colunas o dataframe deve possuir
dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)
filtro = dfBruto["<TIME>"] != "18:00:00"        # filtra apenas as velas das 9h às 17h
dfTratado = dfBruto[filtro].loc[:, ["<DATE>", "<TIME>"]].copy()     # copia para novo dataframe apenas as colunas "<DATE>" e "<TIME>", mantendo os mesos índices

# -----verificando se o dataframe contem todos as velas 9h-17h) do dia-------------
condicao = dfTratado['<DATE>'].value_counts() < 9                       # TRUE para dias que tem menos de 8 velas (das 9h às 17h)
datas = ((dfTratado['<DATE>'].value_counts()[condicao]).index).values   # datas dos dias incompletos
print('As seguintes datas estão incompletas e foram apagadas:')
for i in range(0, len(datas)):                                          # apagando o dia incompleto
    print(f"{datas[i]}")                                                # mostra a data excluida do dftratado
    datasIndice = ((dfTratado['<DATE>']) == datas[i])                   # recebe um dataframe com 'True' nas linhas do dia comparado com a data
    lista = dfTratado[datasIndice]                         # recebe um dataframe com as linhas do dia comparado
    listaIndice = lista.index  # recebe uma lista com os índices do dataframe 'lista'
    dfTratado.drop(listaIndice, inplace=True)        # apaga as linhas dos dias incompletos
# ---------------------------------------------------------

#dfTratado.to_csv("dfTratado.csv", index=False)      # salva como csv, sem os índices

#-----Criando novas colunas-------------------------
dfTratado['<TAMANHO>'] = np.nan                                 # tam (tamanho da vela)
dfTratado['<TAMANHO NORMALIZADO>'] = np.nan                     # tamNor (normalização do tamanho da vela)
dfTratado['<VARIAÇÃO DO PREÇO>'] = np.nan                       # var (variação de preço)
dfTratado['<VARIAÇÃO DO PREÇO NORMALIZADO>'] = np.nan           # varNor (normalização da variação de preço)
dfTratado['<PREÇO>'] = np.nan                                   # preco (valor de abertura normalizado)
dfTratado['<VOLUME NORMALIZADO>'] = np.nan                      # volNor (volume normalizado)
dfTratado['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>'] = np.nan     # tamMed (é o tamanho médio entre as velas do dia)
dfTratado['<VOLUME MÉDIO NORMALIZADO DAS VELAS>'] = np.nan      # volMed (é o volume médio normalizado entre as velas do dia)
dfTratado['<OPERAÇÃO>'] = np.nan                                # Compra, Venda ou lateral
#-----------------------------------------------


#----------------Tratando o dataframe------------------
dfTratado['<TAMANHO>'] = dfBruto['<CLOSE>'] - dfBruto['<OPEN>']       # tam (tamanho da vela)
dfTratado['<VARIAÇÃO DO PREÇO>'] = dfBruto['<HIGH>'] - dfBruto['<LOW>']  # var (variação de preço)
somaTamanho = 0
for indice, coluna in dfTratado.iterrows():
    if (coluna["<TIME>"] == "09:00:00"):    # para as velas das 9h às 16h
        varMax = float(dfBruto.loc[indice:indice+7, ["<HIGH>"]].max().values)    # maior preço de negociação no dia (das 9h às 16h)
        ivol = float(dfBruto.loc[indice:indice+7, ["<VOL>"]].max().values)# (maior volume encontrado no dia, das 9h às 16h)
        ivar = int(dfTratado.loc[indice:indice + 7,["<VARIAÇÃO DO PREÇO>"]].max().values)  # maior variação de preço do dia das 9h às 16h

        dfTratado.loc[indice:indice+8, ["<TAMANHO NORMALIZADO>"]] = ((dfTratado.loc[indice:indice+8, ["<TAMANHO>"]]).values) / ivar  # tamNor (normalização do tamanho da vela)
        dfTratado.loc[indice:indice+8, ["<VARIAÇÃO DO PREÇO NORMALIZADO>"]] = (((dfTratado.loc[indice:indice+8, ["<VARIAÇÃO DO PREÇO>"]]).values)) / ivar  # varNor (normalização da variação de preço)
        dfTratado.loc[indice:indice+8, ['<PREÇO>']] = (((dfBruto.loc[indice:indice+8, ["<OPEN>"]]).values)) / varMax  # preco (valor de abertura normalizado)
        dfTratado.loc[indice:indice+8, ['<VOLUME NORMALIZADO>']] = (((dfBruto.loc[indice:indice+8, ["<VOL>"]]).values)) / ivol  # volNor (volume normalizado)

        somaTamanho = abs(dfTratado.loc[indice:indice+7, ["<TAMANHO NORMALIZADO>"]].values).sum()   # soma o valor absoluto do tamanho normalizado das velas das 8h às 16h
        somaVolume = abs(dfTratado.loc[indice:indice+7, ["<VOLUME NORMALIZADO>"]].values).sum()     # soma o valor absoluto do volume normalizado das velas das 8h às 16h
        dfTratado.loc[[indice+8], ['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>']] = somaTamanho / 8      # tamMed (é o tamanho médio entre as velas do dia)
        dfTratado.loc[[indice+8], ['<VOLUME MÉDIO NORMALIZADO DAS VELAS>']] = somaVolume / 8        # volMed (é o volume médio normalizado entre as velas do dia)


        #VARIÁVEIS TRATADAS PARA TREINO (COMPRA)
        if( ((int(dfTratado.loc[[indice+8], ["<TAMANHO>"]].values)) > 0) and ((abs(float(dfTratado.loc[[indice+8],["<TAMANHO NORMALIZADO>"]].values))) >= (float(dfTratado.loc[[indice+8],['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>']].values))) and ((float(dfTratado.loc[[indice+8],["<VOLUME NORMALIZADO>"]].values)) >= (float(dfTratado.loc[[indice+8],['<VOLUME MÉDIO NORMALIZADO DAS VELAS>']].values)*0.5)) ):
            dfTratado.loc[[indice+8], ['<OPERAÇÃO>']] = 'COMPRA'

        # VARIÁVEIS TRATADAS PARA TREINO (venda)
        elif( ((int(dfTratado.loc[[indice+8], ["<TAMANHO>"]].values)) < 0) and ((abs(float(dfTratado.loc[[indice+8],["<TAMANHO NORMALIZADO>"]].values))) >= (float(dfTratado.loc[[indice+8],['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>']].values))) and ((float(dfTratado.loc[[indice+8],["<VOLUME NORMALIZADO>"]].values)) >= (float(dfTratado.loc[[indice+8],['<VOLUME MÉDIO NORMALIZADO DAS VELAS>']].values)*0.5)) ):
            dfTratado.loc[[indice+8], ['<OPERAÇÃO>']] = 'VENDA'

        # VARIÁVEIS TRATADAS PARA TREINO (LATERAL)
        else:
            dfTratado.loc[[indice+8],['<OPERAÇÃO>']] = 'LATERAL'
#----------------------------------------------------

'''#------mostra quantas saidas foram geradas------
filtroCompra = dfTratado['<OPERAÇÃO>'] == 'COMPRA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
print(f"QUANTIDADE DE COMPRAS: {sum(filtroCompra * 1)}")          # mostra quantas linhas que contém o valor "John"
filtroVenda = dfTratado['<OPERAÇÃO>'] == 'VENDA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
print(f"QUANTIDADE DE VENDA: {sum(filtroVenda * 1)}")          # mostra quantas linhas que contém o valor "John"
filtroLateral = dfTratado['<OPERAÇÃO>'] == 'LATERAL'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
print(f"QUANTIDADE DE LATERAL: {sum(filtroLateral * 1)}")          # mostra quantas linhas que contém o valor "John"
#---------------------------------------------
'''
#dfTratado.to_excel("dfTratado.xlsx", index=False)      # salva como csv, sem os índices
#-----Novo dataframe apenas com as linha últeis---------------------
colunasDf = ["<TIME>", "<PREÇO>", "<TAMANHO NORMALIZADO>", "<VARIAÇÃO DO PREÇO NORMALIZADO>", "<VOLUME NORMALIZADO>", "<OPERAÇÃO>"] # seleciona quais colunas o dataframe deve possuir
dfLimpo = dfTratado[colunasDf]
#dfLimpo.to_excel("dfLimpo.xlsx")      # salva como csv, sem os índices
#--------------------------------------------------------------------


#-----Gerando o dataframe para a RNA---------
'''dfRna = pd.DataFrame({
    'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],
    'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],
    'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],
    'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0],'X':[0]
})'''
#for i in range(0,8):
dfRna = dfLimpo.loc[0]
dfRna.drop(["<TIME>","<OPERAÇÃO>"], inplace=True)    # o 'inplace=True' retorna a alteração para o próprio dataframe
dfRna.loc[1] = dfLimpo.loc[1]

'''print(dfRna)
new = 0
contLinha = 0
for linha, dado in dfLimpo.iterrows():
    listaLinha = (dado[1:].values)  # recebe uma lista em float com os valores da linha atual do dataframe
    if(new == 0):
        if (dado["<TIME>"] != "17:00:00"):
            for i in range(0,4):
                dfRna.loc[linha,[contLinha]] = listaLinha[i]
                contLinha = contLinha + 1
        else:
            for i in range(0,5):
                dfRna.loc[[linha],[contLinha]] = listaLinha[i]
'''


print((dfRna))

dfRna.to_excel("dfRna.xlsx")      # salva como csv, sem os índices




