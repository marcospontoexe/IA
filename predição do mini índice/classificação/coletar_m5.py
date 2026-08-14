"""
Recolha do histórico completo de M5 do WIN$N via API do MetaTrader 5.

Contorna o limite de ~100.000 barras da exportação da interface, descarregando
em blocos anuais e concatenando.

PRÉ-REQUISITOS
  - MetaTrader 5 instalado, ABERTO e com a conta ligada (a API fala com o terminal).
  - pip install MetaTrader5 pytz pandas
  - Python <= 3.13 no Windows (o pacote MetaTrader5 não tem wheel para 3.14).
    Use o mesmo interpretador do kernel do notebook (Python 3.13.9), NÃO o python
    do PATH do sistema.

USO
  python coletar_m5.py
  (ou cole o conteúdo numa célula do notebook)

SAÍDA
  WIN$N_M5_BRUTO_COMPLETO.csv  no mesmo formato dos ficheiros existentes:
  datetime;Abertura;Fechamento;Maxima;Minima;Volume_Real;Volume_Tick
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("MetaTrader5 não instalado. Execute: pip install MetaTrader5")

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------
SIMBOLO = "WIN$N"
TIMEFRAME = mt5.TIMEFRAME_M5
ANO_INICIO = 2020          # antes do início real da série; blocos vazios são ignorados
ANO_FIM = 2026
SAIDA = "WIN$N_M5_BRUTO_COMPLETO.csv"
REFERENCIA = "WIN$N_M5_BRUTO.csv"   # ficheiro antigo, usado só para validar o fuso

# A API do MT5 interpreta datetimes como UTC. Construir as fronteiras em UTC
# evita que o Python aplique o offset do fuso local e desloque os blocos.
UTC = pytz.timezone("Etc/UTC")

COLUNAS = {
    "open": "Abertura",
    "close": "Fechamento",
    "high": "Maxima",
    "low": "Minima",
    "real_volume": "Volume_Real",
    "tick_volume": "Volume_Tick",
}


def conectar():
    if not mt5.initialize():
        sys.exit(f"Falha ao inicializar o MT5: {mt5.last_error()}\n"
                 "Confirme que o terminal está aberto e com a conta ligada.")

    info = mt5.terminal_info()
    print(f"Terminal: {info.name} build {mt5.version()[1]}")
    print(f"Máx. barras no gráfico (maxbars): {info.maxbars:,}")
    if info.maxbars <= 100_000:
        print("  AVISO: maxbars baixo. Em Ferramentas > Opções > Gráficos, defina")
        print("  'Máx. barras no gráfico' como Ilimitado e reinicie o MT5.")
        print("  A API pode devolver menos barras do que o pedido enquanto isto não for feito.")

    if not mt5.symbol_select(SIMBOLO, True):
        mt5.shutdown()
        sys.exit(f"Símbolo {SIMBOLO} indisponível: {mt5.last_error()}")


def descarregar_blocos():
    """Descarrega ano a ano. Blocos vazios (antes do início da série) são ignorados."""
    partes = []
    for ano in range(ANO_INICIO, ANO_FIM + 1):
        inicio = UTC.localize(datetime(ano, 1, 1))
        fim = UTC.localize(datetime(ano + 1, 1, 1))

        rates = mt5.copy_rates_range(SIMBOLO, TIMEFRAME, inicio, fim)
        n = 0 if rates is None else len(rates)
        print(f"  {ano}: {n:>7,} barras" + ("" if n else f"   ({mt5.last_error()})"))

        if n:
            partes.append(pd.DataFrame(rates))

    if not partes:
        mt5.shutdown()
        sys.exit("Nenhuma barra devolvida. Force o download abrindo o gráfico "
                 f"{SIMBOLO} em M5 e pressionando Home até parar de carregar.")

    return partes


def montar(partes):
    df = pd.concat(partes, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = (df.drop_duplicates(subset="datetime")
            .sort_values("datetime")
            .set_index("datetime"))
    df = df[list(COLUNAS.keys())].rename(columns=COLUNAS)
    return df


def validar_fuso(df):
    """
    Compara timestamps sobrepostos com o ficheiro antigo. Se houver desvio de fuso,
    os preços de abertura não vão coincidir e o desvio é reportado em horas.
    """
    if not os.path.exists(REFERENCIA):
        print("  (ficheiro de referência ausente — validação de fuso ignorada)")
        return

    velho = pd.read_csv(REFERENCIA, sep=";", parse_dates=["datetime"], index_col="datetime")
    comum = df.index.intersection(velho.index)
    if len(comum) == 0:
        print("  AVISO: nenhum timestamp em comum com o ficheiro antigo — verifique o fuso.")
        return

    iguais = np.isclose(df.loc[comum, "Abertura"], velho.loc[comum, "Abertura"]).mean()
    print(f"  {len(comum):,} timestamps em comum | aberturas coincidentes: {iguais:.1%}")
    if iguais < 0.99:
        print("  AVISO: os preços não coincidem nos mesmos timestamps.")
        print("  Provável desvio de fuso horário entre a API e a exportação da interface.")
        print("  Compare o horário de pregão impresso abaixo com o do ficheiro antigo.")


def diagnostico(df):
    h = df.index.hour + df.index.minute / 60
    dias = pd.Series(df.index.date).nunique()

    feat = df.between_time("09:00", "10:59")
    lab = df.between_time("11:00", "11:59")
    cf = feat.groupby(feat.index.date).size()
    cl = lab.groupby(lab.index.date).size()
    uteis = len(set(cf[cf >= 22].index) & set(cl[cl >= 11].index))

    print(f"\n  Período .............. {df.index.min():%Y-%m-%d} a {df.index.max():%Y-%m-%d}")
    print(f"  Barras ............... {len(df):,}")
    print(f"  Dias distintos ....... {dias:,}")
    print(f"  Pregão ............... {h.min():.2f}h a {h.max():.2f}h")
    print(f"  AMOSTRAS ÚTEIS ....... {uteis:,}   (janelas 09:00-10:59 e 11:00-11:59 completas)")
    return uteis


if __name__ == "__main__":
    print("=== RECOLHA DE HISTÓRICO M5 ===\n")
    conectar()

    print("\nDescarregando blocos anuais...")
    partes = descarregar_blocos()
    mt5.shutdown()

    df = montar(partes)

    print("\nValidando fuso horário contra o ficheiro existente...")
    validar_fuso(df)

    print("\nDiagnóstico do resultado:")
    uteis = diagnostico(df)

    df.to_csv(SAIDA, sep=";", encoding="utf-8")
    print(f"\nGravado: {SAIDA}")

    if uteis < 1000:
        print("\nAVISO: menos de 1.000 amostras úteis. O terminal provavelmente ainda")
        print("não tem todo o histórico em cache. Abra o gráfico WIN$N em M5, pressione")
        print("Home até o gráfico parar de carregar, e execute este script de novo.")
    else:
        print(f"\nOK: {uteis:,} amostras úteis. Substitua WIN$N_M5_BRUTO.csv por este ficheiro")
        print("no pipeline, ou aponte NOME_ARQUIVO_ENTRADA para ele.")
