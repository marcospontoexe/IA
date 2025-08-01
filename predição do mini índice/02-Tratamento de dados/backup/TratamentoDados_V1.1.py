import pandas as pd
import numpy as np

selecionarCol = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] # seleciona quais colunas o dataframe deve possuir
dfBruto = pd.read_csv("WIN$_H1.csv", sep="\t", usecols=selecionarCol)
filtro = dfBruto["<TIME>"] != "18:00:00"        # filtra apenas as velas das 9h às 17h
dfTratado = dfBruto[filtro].loc[:, ["<DATE>", "<TIME>"]].copy()     # copia para novo dataframe apenas as colunas "<DATE>" e "<TIME>", mantendo os mesos índices
#dfTratado.drop([8], inplace=True)
#condicao = dfTratado['<DATE>'].unique().values
print(dfTratado['<DATE>'].value_counts())
#-----verificando se o dataframe contem todos as velas 9h-17h) do dia-------------
flag = 0
cont = 0
for indice, coluna in dfTratado.iterrows():     # for para percorrer todas as linhas do dataframe
    if(flag == 0):
        temp = coluna['<DATE>']
        hora = 9
        flag = 1
        lista=[]


    if( (coluna['<DATE>'] == temp) and (coluna['<TIME>'] == (f"{hora:>02}:00:00")) ):   # verifica se o dia "<DATE>" contem todas as velas (das 9h às 17h)
        cont = cont + 1
        if (hora == 17):
            flag = 0
    else:               # quando falta alguma vela no dia verificado
        if(coluna['<DATE>'] == temp):
            lista.append(int((dfTratado.loc[[indice]].index).values))   # adiciona qual foi o índice onde ocorreu o erro

        else:    # quando muda o dia e
            print(f"ERRO nasseguintes linhas;\n{dfTratado.loc[lista]}")             # mostra o dia, o horário de cada vela, e o índice no dataframe onde está inconpleto
            print('--------------------------------------------')
            print(f"AS VELAS DAS {str((dfTratado.loc[lista,['<TIME>']]).values)} HORAS\nDO DIA {temp}\nFORAM APAGADOS DO DATAFRAME!")   # mostra quias velas foram apagadas do dataframe
            print('--------------------------------------------')
            dfTratado.drop(lista, inplace=True)  # exclui as linhas dos dias incompletos (9-17h)
            temp = coluna['<DATE>']                             # recebe a data atual a ser verificada
            hora = 9
            lista = []
        #exit()      # para o programa
    hora = hora + 1
#---------------------------------------------------------

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
dfTratado.to_excel("dfTratado.xlsx", index=False)      # salva como csv, sem os índices

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


