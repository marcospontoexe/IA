import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




selecionarCol = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] # seleciona quais colunas o dataframe deve possuir
dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)
filtro = dfBruto["<TIME>"] != "18:00:00"        # filtra apenas as velas das 9h às 17h
dfTratado = dfBruto[filtro].loc[:, ["<DATE>", "<TIME>"]].copy()     # copia para novo dataframe apenas as colunas "<DATE>" e "<TIME>", mantendo os mesos índices



# -----verificando se o dataframe contem todos as velas 9h-17h) do dia-------------
condicao = dfTratado['<DATE>'].value_counts() < 9                       # TRUE para dias que tem menos de 9 velas (das 9h às 17h)
datas = ((dfTratado['<DATE>'].value_counts()[condicao]).index).values   # datas dos dias incompletos
print('As seguintes datas estão incompletas e foram apagadas:')
for i in range(0, len(datas)):                                          # apagando o dia incompleto
    print(f"{datas[i]}")                                                # mostra a data excluida do dftratado
    datasIndice = ((dfTratado['<DATE>']) == datas[i])                   # recebe um dataframe com 'True' nas linhas do dia comparado com a data
    lista = dfTratado[datasIndice]                         # recebe um dataframe com as linhas do dia comparado
    listaIndice = lista.index  # recebe uma lista com os índices do dataframe 'lista'
    dfTratado.drop(listaIndice, inplace=True)        # apaga as linhas dos dias incompletos
# ---------------------------------------------------------


#-----Criando novas colunas-------------------------
dfTratado['<TAMANHO>'] = np.nan                                 # tam (tamanho da vela)
dfTratado['<TAMANHO NORMALIZADO>'] = np.nan                     # tamNor (normalização do tamanho da vela)
dfTratado['<VARIAÇÃO DO PREÇO>'] = np.nan                       # var (variação de preço)
dfTratado['<VARIAÇÃO DO PREÇO NORMALIZADO>'] = np.nan           # varNor (normalização da variação de preço)
dfTratado['<PREÇO>'] = np.nan                                   # preco (valor de abertura normalizado)
dfTratado['<VOLUME NORMALIZADO>'] = np.nan                      # volNor (volume normalizado)
dfTratado['<PREÇO MÉDIO NORMALIZADO DAS VELAS>'] = np.nan       # méia entre as velas das 9h até 16h
dfTratado['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>'] = np.nan     # méia entre as velas das 9h até 16h
dfTratado['<VARIAÇÃO MÉDIO NORMALIZADO DAS VELAS>'] = np.nan     # méia entre as velas das 9h até 16h
dfTratado['<VOLUME MÉDIO NORMALIZADO DAS VELAS>'] = np.nan      # méia entre as velas das 9h até 16h
dfTratado['<OPERAÇÃO>'] = np.nan                                # Compra, Venda ou lateral
#-----------------------------------------------


#----------------Tratando o dataframe------------------
dfTratado['<TAMANHO>'] = dfBruto['<CLOSE>'] - dfBruto['<OPEN>']       # tam (tamanho da vela)
dfTratado['<VARIAÇÃO DO PREÇO>'] = dfBruto['<HIGH>'] - dfBruto['<LOW>']  # var (variação de preço)
somaTamanho = 0
for indice, coluna in dfTratado.iterrows():
    if (coluna["<TIME>"] == "09:00:00"):    # para as velas das 9h às 16h
        valorMax = float(dfBruto.loc[indice:indice + 7, ["<HIGH>"]].max().values)  # maior preço de negociação no dia (das 9h às 16h)
        valorMin = float(dfBruto.loc[indice:indice + 7, ["<LOW>"]].min().values)  # menor preço de negociação no dia (das 9h às 16h)
        volMax = float(dfBruto.loc[indice:indice+7, ["<VOL>"]].max().values)# (maior volume encontrado no dia, das 9h às 16h)
        volMin = float(dfBruto.loc[indice:indice+7, ["<VOL>"]].min().values)  # (menor volume encontrado no dia, das 9h às 16h)
        varMax = int(dfTratado.loc[indice:indice+7,["<VARIAÇÃO DO PREÇO>"]].max().values)  # maior variação de preço do dia das 9h às 16h
        varMin = int(dfTratado.loc[indice:indice+7, ["<VARIAÇÃO DO PREÇO>"]].min().values)  # menor variação de preço do dia das 9h às 16h
        tamMax = int(abs(dfTratado.loc[indice:indice+7, ["<TAMANHO>"]]).max().values)  # maior tamanho de vela do dia das 9h às 16h
        tamMin = int(abs(dfTratado.loc[indice:indice+7,["<TAMANHO>"]]).min().values)  # menor tamanho de vela do dia das 9h às 16h

        dfTratado.loc[indice:indice+8, ["<TAMANHO NORMALIZADO>"]] = ((dfTratado.loc[indice:indice+8, ["<TAMANHO>"]]).values)  / tamMax # tamNor (normalização do tamanho da vela)
        dfTratado.loc[indice:indice+8, ["<VARIAÇÃO DO PREÇO NORMALIZADO>"]] = ((((dfTratado.loc[indice:indice+8, ["<VARIAÇÃO DO PREÇO>"]]).values))-varMin) / (varMax - varMin)  # varNor (normalização da variação de preço)
        dfTratado.loc[indice:indice+8, ['<PREÇO>']] = (((dfBruto.loc[indice:indice+8, ["<OPEN>"]]).values) - valorMin) / (valorMax - valorMin) # preco (valor de abertura normalizado)
        dfTratado.loc[indice:indice+8, ['<VOLUME NORMALIZADO>']] = ((((dfBruto.loc[indice:indice+8, ["<VOL>"]]).values)) - volMin)/ (volMax - volMin) # volNor (volume normalizado)

        somaPreco = abs(dfTratado.loc[indice:indice+7,['<PREÇO>']].values).sum()                    # soma o valor absoluto do preço das velas das 9h às 16h
        somaVariacao = abs(dfTratado.loc[indice:indice+7,["<VARIAÇÃO DO PREÇO NORMALIZADO>"]].values).sum()    # soma o valor absoluto da variação de preço das velas das 9h às 16h
        somaTamanho = abs(dfTratado.loc[indice:indice+7, ["<TAMANHO NORMALIZADO>"]].values).sum()   # soma o valor absoluto do tamanho normalizado das velas das 9h às 16h
        somaVolume = abs(dfTratado.loc[indice:indice+7, ["<VOLUME NORMALIZADO>"]].values).sum()     # soma o valor absoluto do volume normalizado das velas das 9h às 16h
        dfTratado.loc[[indice+8],['<PREÇO MÉDIO NORMALIZADO DAS VELAS>']] = somaPreco / 8  # tamMed (é o tamanho médio normalizado entre as velas das 9h às 16h)
        dfTratado.loc[[indice+8], ['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>']] = somaTamanho / 8      # tamMed (é o tamanho médio entre as velas das 9h às 16h)
        dfTratado.loc[[indice+8], ['<VARIAÇÃO MÉDIO NORMALIZADO DAS VELAS>']] = somaVariacao / 8        # volMed (é o volume médio normalizado entre as velas das 9h às 16h)
        dfTratado.loc[[indice+8],['<VOLUME MÉDIO NORMALIZADO DAS VELAS>']] = somaVolume / 8  # volMed (é o volume médio normalizado entre as velas das 9h às 16h)

        #VARIÁVEIS TRATADAS PARA TREINO (COMPRA)
        if( ((int(dfTratado.loc[[indice+8], ["<TAMANHO>"]].values)) > 0) and ((abs(float(dfTratado.loc[[indice+8],["<TAMANHO NORMALIZADO>"]].values))) >= (float(dfTratado.loc[[indice+8],['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>']].values)*1)) ):
            dfTratado.loc[[indice+8], ['<OPERAÇÃO>']] = 'COMPRA'

        # VARIÁVEIS TRATADAS PARA TREINO (venda)
        elif( ((int(dfTratado.loc[[indice+8], ["<TAMANHO>"]].values)) < 0) and ((abs(float(dfTratado.loc[[indice+8],["<TAMANHO NORMALIZADO>"]].values))) >= (float(dfTratado.loc[[indice+8],['<TAMANHO MÉDIO NORMALIZADO DAS VELAS>']].values)*1)) ):
            dfTratado.loc[[indice+8], ['<OPERAÇÃO>']] = 'VENDA'

        # VARIÁVEIS TRATADAS PARA TREINO (LATERAL)
        else:
            dfTratado.loc[[indice+8],['<OPERAÇÃO>']] = 'LATERAL'
#----------------------------------------------------


#------mostra quantas saidas foram geradas------
print(f"Quantidade de rótulos antes da equalização:")
filtroCompra = dfTratado['<OPERAÇÃO>'] == 'COMPRA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
compra = sum(filtroCompra * 1)
print(f"QUANTIDADE DE COMPRAS: {compra}")          # mostra quantas linhas que contém o valor "John"
filtroVenda = dfTratado['<OPERAÇÃO>'] == 'VENDA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
venda = sum(filtroVenda * 1)
print(f"QUANTIDADE DE VENDA: {venda}")          # mostra quantas linhas que contém o valor "John"
filtroLateral = dfTratado['<OPERAÇÃO>'] == 'LATERAL'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
lateral = sum(filtroLateral * 1)
print(f"QUANTIDADE DE LATERAL: {lateral}")          # mostra quantas linhas que contém o valor "John"
menor = min(compra, venda, lateral)
print(f"menor quantidade: {menor}")
#---------------------------------------------



#----Mantem proporção de valores entre as classes------
if(compra > menor):     # caso a quantidadede 'COMPRA' não seja o menor valor
    datasCompra = dfTratado[filtroCompra]['<DATE>'].values      # array contendo as datas de '<OPERAÇÃO>' == 'COMPRA'
    difer = compra-menor
    for i in range(0,difer):
        indiceCompras = ((dfTratado[dfTratado['<DATE>'] == datasCompra[i]]).index).values   # array com os índices de cada data
        dfTratado.drop(indiceCompras, inplace=True)             # apaga os índices das datas 'datasCompra'


if(venda > menor):     # caso a quantidadede 'VENDE' não seja o menor valor
    datasVenda = dfTratado[filtroVenda]['<DATE>'].values      # array contendo as datas de '<OPERAÇÃO>' == 'VENDA'
    difer = venda-menor
    for i in range(0,difer):
        indiceVendas = ((dfTratado[dfTratado['<DATE>'] == datasVenda[i]]).index).values
        dfTratado.drop(indiceVendas, inplace=True)


if(lateral > menor):         # caso a quantidadede 'LATERAL' não seja o menor valor
    datasLateral = dfTratado[filtroLateral]['<DATE>'].values  # array contendo as datas de '<OPERAÇÃO>' == 'LATERAL'
    difer = lateral - menor
    for i in range(0, difer):
        indiceLateral = ((dfTratado[dfTratado['<DATE>'] == datasLateral[i]]).index).values
        dfTratado.drop(indiceLateral, inplace=True)
#------------------------------------------------------

#------mostra quantas saidas foram geradas------
print(f"Quantidade de rótulos após a equalização:")
filtroCompra = dfTratado['<OPERAÇÃO>'] == 'COMPRA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
compra = sum(filtroCompra * 1)
print(f"QUANTIDADE DE COMPRAS: {compra}")          # mostra quantas linhas que contém o valor "John"
filtroVenda = dfTratado['<OPERAÇÃO>'] == 'VENDA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
venda = sum(filtroVenda * 1)
print(f"QUANTIDADE DE VENDA: {venda}")          # mostra quantas linhas que contém o valor "John"
filtroLateral = dfTratado['<OPERAÇÃO>'] == 'LATERAL'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
lateral = sum(filtroLateral * 1)
print(f"QUANTIDADE DE LATERAL: {lateral}")          # mostra quantas linhas que contém o valor "John"
menor = min(compra, venda, lateral)
#---------------------------------------------


#------reorganização do dataframe-------------
dfReordenado = dfTratado.copy()
tam = dfTratado.shape[0]    # quantidade de linhas do dftratado
filtroCompra1 = dfTratado['<OPERAÇÃO>'] == 'COMPRA'   # recebe uma série contendo, "True" quando os valores da coluna "<OPERAÇÃO>" é "COMPRA", e "False" caso contrário
filtroVenda1 = dfTratado['<OPERAÇÃO>'] == 'VENDA'
filtroLateral1 = dfTratado['<OPERAÇÃO>'] == 'LATERAL'
datasCompra1 = dfTratado[filtroCompra1]['<DATE>'].values      # array contendo as datas de '<OPERAÇÃO>' == 'COMPRA'
datasVenda1 = dfTratado[filtroVenda1]['<DATE>'].values      # array contendo as datas de '<OPERAÇÃO>' == 'COMPRA'
datasLateral1 = dfTratado[filtroLateral1]['<DATE>'].values      # array contendo as datas de '<OPERAÇÃO>' == 'COMPRA'
indiceDfReordenado = dfReordenado.index.values
cont = 0
for i in range(0, len(datasCompra1) ):
    indiceCompras1 = ((dfTratado[dfTratado['<DATE>'] == datasCompra1[i]]).index).values  # array com os índices de cada data
    indiceVendas1 = ((dfTratado[dfTratado['<DATE>'] == datasVenda1[i]]).index).values  # array com os índices de cada data
    indiceLateral1 = ((dfTratado[dfTratado['<DATE>'] == datasLateral1[i]]).index).values  # array com os índices de cada data
    for j in range(0, len(indiceCompras1)):
        dfReordenado.loc[[indiceDfReordenado[cont]]] = dfTratado.loc[[indiceCompras1[j]]].values
        cont = cont+1
    for j in range(0, len(indiceVendas1)):
        dfReordenado.loc[[indiceDfReordenado[cont]]] = dfTratado.loc[[indiceVendas1[j]]].values
        cont = cont+1
    for j in range(0, len(indiceLateral1)):
        dfReordenado.loc[[indiceDfReordenado[cont]]] = dfTratado.loc[[indiceLateral1[j]]].values
        cont = cont+1
#---------------------------------------------

dfTratado.to_excel("dfTratado.xlsx")      # salva como csv, sem os índices         #############################################
dfLimpo = dfReordenado.loc[:,["<DATE>", "<TIME>", '<PREÇO MÉDIO NORMALIZADO DAS VELAS>', '<TAMANHO MÉDIO NORMALIZADO DAS VELAS>', '<VARIAÇÃO MÉDIO NORMALIZADO DAS VELAS>', '<VOLUME MÉDIO NORMALIZADO DAS VELAS>', '<OPERAÇÃO>']].copy()

#-----removendo linhas do dataframe que possuem valor NaN
dfLimpo.dropna(inplace=True)
#--------------------------------------------------------------------

'''#--------visualizando o dataframe----------
plt.boxplot(dfTratado["<VOLUME NORMALIZADO>"])
#plt.title("PREÇO DA VELA")
plt.xlabel("dias")
plt.ylabel("VALOR")
plt.show()
#-----------------------------------------
'''
#-----verificando a correlação--------
cor = dfLimpo.loc[:,['<PREÇO MÉDIO NORMALIZADO DAS VELAS>', '<TAMANHO MÉDIO NORMALIZADO DAS VELAS>', '<VARIAÇÃO MÉDIO NORMALIZADO DAS VELAS>', '<VOLUME MÉDIO NORMALIZADO DAS VELAS>']].corr()
#cor.style.background_gradient(cmap='coolwarm')
print(cor)
cor.to_excel("correlação.xlsx")      # salva como csv, sem os índices      ##################################################
#--------------------------------------

#-----Gerando o dataframe para a RNA---------
dfRna = pd.DataFrame({          # cria um dataframe com apenas uma linha
    'X1':[1],'X2':[1],'X3':[1],'X4':[1],'Y1':[1],'Y2':[1],'Y3':[1]
})
contColuna = 0  # iteração para incluir colunas no dataframe 'dfRNA'
contLinha = 0   # iteração para incluir linhas no dataframe 'dfRNA'
for linha, dado in dfLimpo.iterrows():
    listaLinha = (dado[2:].values)  # recebe uma lista em float com os valores da linha atual do dataframe (sem a coluna '<DATE>' e '<TIME>')

    for i in range(0,4):
        contColuna = contColuna + 1
        dfRna.loc[contLinha,[f"X{contColuna}"]] = listaLinha[i]

   
    if(listaLinha[4] == "COMPRA"):
        dfRna.loc[[contLinha], ["Y1"]] = 1
        dfRna.loc[[contLinha], ["Y2"]] = 0
        dfRna.loc[[contLinha], ["Y3"]] = 0
    elif(listaLinha[4] == "LATERAL"):
        dfRna.loc[[contLinha], ["Y1"]] = 0
        dfRna.loc[[contLinha], ["Y2"]] = 1
        dfRna.loc[[contLinha], ["Y3"]] = 0
    elif(listaLinha[4] == "VENDA"):
        dfRna.loc[[contLinha], ["Y1"]] = 0
        dfRna.loc[[contLinha], ["Y2"]] = 0
        dfRna.loc[[contLinha], ["Y3"]] = 1
    else:
        print(f'ERRO: {listaLinha[4]}')

    contColuna = 0
    contLinha = contLinha + 1

#print((dfRna))

dfBruto.to_excel("dfBruto.xlsx")                                #############################################################
dfLimpo.to_excel("dfLimpo.xlsx")      # salva como csv, sem os índices      ###########################################
cor.to_excel("correlação.xlsx")      # salva como csv, sem os índices      ##################################################
dfRna.to_excel("dfRna.xlsx", index=False)      # salva sem os índices
dfRna.to_csv("dfRna.csv", index=False)      # salva sem os índices
dfReordenado.to_excel("dfReordenado.xlsx")      # salva sem os índices








