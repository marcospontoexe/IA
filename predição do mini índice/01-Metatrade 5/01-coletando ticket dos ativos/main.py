import MetaTrader5 as mt5       #importa a biblioteca

mt5.initialize()        #inicia o metatrader 5

tickets  = mt5.symbols_total()      # recebe os tickets dos ativos
