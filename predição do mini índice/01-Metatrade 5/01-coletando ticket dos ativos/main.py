import MetaTrader5 as mt5       #importa a biblioteca

mt5.initialize()        #inicia o metatrader 5

ticketsTotal  = mt5.symbols_total()      # recebe a quantidade de tickets dos ativos

if ticketsTotal > 0:
    print("Quantidade total de tickets de ativos: ", ticketsTotal)
else:
    print("Síbolos não encontrados!")

ticket = mt5.symbols_get()      #recebe o ticket dos ativos
cont = 0
for i in ticket:
    if cont < 100:
        print("{}: {}".format(cont, i.name))    # imprime o ticket dos ativos

    else:
        break
    cont += 1

mt5.shutdown()      # encerra o metatrader
