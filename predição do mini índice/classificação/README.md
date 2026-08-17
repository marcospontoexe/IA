# Previsão do mini-índice (WIN$N) com machine learning

Este repositório é o meu caderno de experimentos tentando prever o comportamento do
mini-índice brasileiro com aprendizado de máquina. A ideia não é construir um robô e sair
operando — é testar, de forma organizada e honesta, **quais abordagens realmente
funcionam e quais só parecem funcionar**.

Vou passar por várias famílias de técnica: classificação, regressão, previsão de
volatilidade e aprendizado por reforço. Cada uma responde a uma pergunta diferente sobre o
mercado, e cada uma tem um jeito próprio de dar errado.

---

## Por onde estou começando

**Classificar a janela das 11:00–11:59 em Compra, Lateral ou Venda**, usando só o que
aconteceu entre 09:00 e 10:59.

📓 **[classificação Compra-Lateral-Venda do WIN M5.ipynb](classificação%20Compra-Lateral-Venda%20do%20WIN%20M5.ipynb)**

Se funcionar, a operação é direta: entro às 11:00 e saio às 12:00 ou quando atingir um alvo determinado. Compra, eu compro.
Venda, eu vendo. Lateral, fico de fora e não pago custo nenhum.

Escolhi esses horários porque eles cercam os dois momentos de maior energia do pregão: a
**abertura às 09:00**, quando o mercado digere de uma vez tudo que aconteceu desde o
fechamento anterior, e a **abertura americana às 11:30**, quando entra volume novo e o
S&P 500 começa a arrastar o Ibovespa junto.

### Status: resultado nulo, agora com o holdout fechado

Não vou enfeitar. Onze modelos, 976 dias de trabalho, 10 sementes nos tabulares:

| Modelo | Acurácia | ±dp | Ganho sobre o baseline |
|---|---|---|---|
| XGBoost | 0,3604 | 0,0061 | +0,0085 |
| Gradient Boosting | 0,3581 | 0,0103 | +0,0063 |
| CNN 1D | 0,3549 | 0,0031 | +0,0008 |
| **Baseline (classe majoritária)** | **0,3519** | 0,0000 | — |
| LSTM | 0,3491 | 0,0053 | −0,0050 |
| Random Forest | 0,3479 | 0,0064 | −0,0040 |
| SVM (RBF) | 0,3469 | 0,0000 | −0,0049 |
| MLP | 0,3416 | 0,0037 | −0,0124 |
| Baseline (estratificado) | 0,3401 | 0,0131 | −0,0117 |
| TCN | 0,3342 | 0,0083 | −0,0199 |
| Regressão logística | 0,3136 | 0,0000 | −0,0383 |

Três modelos ficam acima do baseline — e **nenhum de forma significativa**. Teste de
permutação com 200 embaralhamentos no XGBoost: **p = 0,144**. Friedman entre os sete
tabulares: **p = 0,463**, e o baseline majoritário termina em **3º de 7**, à frente de
quatro modelos reais. Nas 21 features, 3 diferem entre classes a p < 0,05 (esperado ~1) e
**nenhuma sobrevive a Bonferroni**. No backtest *out-of-fold*, com custo medido de 7 pontos
por operação: **−8.344 pontos (−R$ 1.668,80)**.

#### O holdout foi aberto — uma vez

Os 244 dias finais (2025-08-18 a 2026-08-12), que nenhuma decisão do projeto encostou:

| | |
|---|---|
| Baseline | 0,3238 |
| Modelo (XGBoost) | 0,3811 |
| **Ganho** | **+5,7 p.p.** |
| Resultado econômico | +5.838 pontos (R$ 1.167,60) |

À primeira vista, o melhor número do projeto. Três testes o desmontaram:

1. **Poder estatístico.** O limiar mínimo detectável, calculado *antes* de abrir, era de
   **8,4 p.p.** O ganho ficou em 5,7 (z = 1,91, p bilateral = 0,056).
2. **Deriva de classe.** A classe majoritária mudou de Venda para Compra entre treino e
   holdout, rebaixando o baseline. Contra a majoritária real do período, o ganho cai para
   **3,7 p.p.** (p = 0,11) — **36% do ganho não era do modelo**.
3. **P&L contra o acaso.** Em 20.000 estratégias aleatórias com as mesmas 161 operações, o
   modelo caiu no **percentil 78** (z = 0,79, p = 0,218). Para dimensionar: no walk-forward,
   a estratégia aleatória rendeu **+44,01 pontos por operação**, contra +36,26 do modelo no
   holdout.

Ou seja: prever direção nessa janela, com essas features, não funciona. É um resultado, e
é por isso que ele está aqui em vez de estar escondido.

> O que me salvou de tratar +5,7 p.p. e R$ 1.167 como descoberta foi ter escrito a regra de
> decisão **antes** de olhar o número. Essa regra está na seção 13 do notebook.

---

## Os dados

| | |
|---|---|
| Ativo | `WIN$N` — mini-índice, rolagem por liquidez, sem ajuste |
| Auxiliar | `WDO$N` — mini-dólar, para features cruzadas |
| Timeframe principal | **M5** |
| Período | 2021-08-12 a 2026-08-12 |
| Pregões | **1.248** |
| Origem | MetaTrader 5 |

O M5 é o ponto ideal. O M1 tem resolução maior mas rende só 121 dias aproveitáveis depois
do limite de exportação do MetaTrader — não dá para treinar nada com isso. O M30 e o H1
rendem 1.242 dias, mas deixam só 4 e 2 velas dentro da janela de entrada, o que mata
qualquer arquitetura sequencial. O M5 fica no meio: 1.240 dias e 24 velas por janela.

> Se o arquivo M5 estiver truncado em 100.000 linhas, é a opção **"Máx. barras no
> gráfico"** do MetaTrader. Coloque em `Ilimitado`, reinicie, aperte `Home` no gráfico até
> parar de carregar e exporte de novo.

---

## As regras que eu sigo

Aprendi na marra que sem isso o resultado fica sempre bonito e sempre falso:

| Regra | Por quê |
|---|---|
| Divisão **sempre cronológica** | Embaralhar série temporal deixa o modelo ver o futuro |
| **Baseline** em todo teste | 35% de acurácia parece bom até você ver que chutar a classe majoritária dá 35% |
| No mínimo **10 sementes** | Uma rodada só mede a sorte da inicialização |
| Scaler ajustado **dentro do fold** | Ajustar na base toda vaza estatística do futuro |
| **Holdout aberto uma vez** | Segunda olhada já é treino disfarçado |
| Resultado em **pontos e reais** | 40% de acerto em 3 classes ainda pode perder dinheiro |
| **Teste de permutação** | É o que separa sinal de sorte |
| **Toda semente declarada** | Sem isso, resultado fraco fica indistinguível de ruído de inicialização |

E uma regra sobre mim mesmo: **a lista de variantes a testar é fechada antes de rodar
qualquer uma**. Testar até achar uma que dá certo e reportar só essa é a forma mais fácil
de se enganar.

---

## O roteiro de experimentos

Cinco abordagens, em ordem de complexidade. Cada uma muda **uma coisa** em relação à
anterior.

### 1. Classificação direcional de período fixo — *em andamento*

Prever a direção de uma janela fixa de tempo. É onde estou.

**Foco:** o *quê* vai acontecer.
**Vantagem:** bem definida e fácil de testar.
**Complexidade:** média.
**Situação:** **resultado nulo, com o holdout fechado.** O ciclo principal terminou —
walk-forward, permutação, Friedman, avaliação econômica e holdout, todos negativos. O que
resta são as variantes pré-registradas: adicionar o dólar (V1), adicionar o futuro do S&P
(V2), mover a janela-alvo para depois das 11:30 (V3), prever volatilidade em vez de direção
(V4) e regressão do retorno (V5). A lista foi fechada antes de rodar qualquer uma.

> Uma ressalva honesta sobre a V3: o holdout foi aberto com a janela de treino
> (`09:00–10:59 → 11:00–11:59`), não com a `09:00–11:59 → 12:00–12:59` que eu tinha
> planejado. A decisão está justificada no notebook — um holdout serve para validar *o mesmo
> modelo* em dado novo, e trocar a janela na hora de abrir misturaria dois efeitos. O custo é
> que **a hipótese das 11:30 perdeu o holdout limpo dela** e só pode ser testada em
> walk-forward daqui em diante.

### 2. Timing de eventos — mudar o foco do "o quê" para o "quando"

Em vez de prever a direção de um período fixo, prever o **momento exato** de uma
oportunidade, com alvo e stop definidos.

O modelo deixa de responder "compra ou venda na próxima hora?" e passa a responder: *"qual
a probabilidade do preço subir X pontos antes de cair Y pontos nos próximos N minutos?"*

Como funciona:
1. **Janela deslizante** — a cada vela nova, o modelo olha as últimas N velas
2. **Saída** — probabilidade de um evento específico: rompimento de máxima, cruzamento de
   média, reversão de RSI
3. **Execução** — se a probabilidade passar de um limiar (75%, por exemplo), entra
4. **Risco** — a ordem já nasce com alvo e stop; sai quando um dos dois bate, e não por
   horário

**Foco:** o *quando*.
**Vantagem:** dinâmica, opera a qualquer hora do dia, e o risco está embutido na
estratégia em vez de ser um detalhe posterior.
**Complexidade:** média-alta.

### 3. Regressão da próxima vela — capturar a memória do mercado

Sair da classificação e ir para regressão: prever o **valor** da próxima vela a partir de
uma sequência das anteriores. É o território natural das LSTMs, que mantêm estado ao longo
do tempo em vez de olhar uma janela fixa.

Como funciona:
1. Treino uma LSTM para, dada uma sequência de 20 velas, prever a 21ª
2. A cada nova vela, alimento a sequência mais recente
3. Se o modelo prevê fechamento significativamente mais alto (> 0,2%, por exemplo), compro
4. Encerro no fim da própria vela, ou seguro enquanto as previsões seguintes confirmarem

**Foco:** o valor da próxima vela.
**Vantagem:** arquitetura desenhada para série temporal; o modelo aprende sozinho a pesar
o recente contra o antigo.
**Complexidade:** alta.

### 4. Previsão de volatilidade — parar de tentar acertar a direção

Talvez a mais promissora, e por um motivo teórico sólido: **volatilidade tem memória,
direção não tem.** A persistência da volatilidade é um dos fatos mais estáveis que existem
em finanças, enquanto o retorno é notoriamente próximo de um passeio aleatório.

Aqui o rótulo deixa de ser Compra/Venda e passa a ser **alta ou baixa volatilidade** para
a janela seguinte.

Como funciona:
1. Rótulo = "alta volatilidade" se o movimento total passar de X pontos, senão "baixa"
2. Se o modelo prevê alta volatilidade, monto uma **estratégia de rompimento**: ordem de
   compra um pouco acima da máxima do período anterior e ordem de venda um pouco abaixo da
   mínima — a primeira que executar cancela a outra
3. Gerencio com alvo ou stop móvel

**Foco:** a magnitude do movimento, não o sentido.
**Vantagem:** não depende de acertar a direção. Lucra com movimento grande para qualquer
lado.
**Complexidade:** média.

### 5. Aprendizado por reforço — aprender a política inteira

A mais avançada, e a que mais se parece com o jeito que um operador humano aprende. O
modelo para de prever e passa a **decidir**.

Como funciona:
1. **Ambiente** — uma simulação do pregão
2. **Agente** — observa o estado do mercado (preços, indicadores, posição atual)
3. **Ação** — a cada vela escolhe entre comprar, vender ou ficar de fora
4. **Recompensa** — lucro gera recompensa positiva, prejuízo gera punição
5. **Aprendizado** — ao longo de milhões de simulações o agente converge para uma
   **política**: um mapeamento de estados de mercado para ações

**Foco:** aprender a estratégia completa, incluindo quando entrar, quando sair e quando
não fazer nada.
**Vantagem:** potencialmente a mais poderosa e adaptativa.
**Desvantagem:** difícil de implementar, difícil de treinar e muito difícil de validar —
é trivial construir um agente que fica rico no simulador e quebra no mercado real.
**Complexidade:** muito alta.

### Resumo comparativo

| Abordagem | Foco principal | Vantagem principal | Complexidade | Situação |
|---|---|---|---|---|
| **1. Classificação direcional** | direção de um período fixo | bem definida, fácil de testar | média | **em andamento** |
| 2. Timing de eventos | o "quando" da oportunidade | dinâmica, risco integrado | média-alta | planejada |
| 3. Regressão da próxima vela | valor da vela seguinte | arquitetura ideal para série temporal | alta | planejada |
| 4. Previsão de volatilidade | magnitude do movimento | não depende de acertar direção | média | planejada |
| 5. Aprendizado por reforço | política de ações ótima | a mais completa e adaptativa | muito alta | planejada |

**Minha ordem de prioridade:** terminar a 1 (inclusive fechando o resultado negativo, se
for esse o caso), pular direto para a **4**, que é a que tem maior chance de dar sinal de
verdade, e depois a **2**, que é o próximo passo mais lógico em termos de gestão de risco.
A 3 e a 5 ficam para quando as anteriores estiverem fechadas.

---

## Estrutura do repositório

```
classificação/
├── classificação Compra-Lateral-Venda do WIN M5.ipynb   ← notebook principal
├── coletar_m5.py                    ← baixa o histórico M5 do MetaTrader
├── WIN$N_M5_BRUTO_COMPLETO.csv      ← base principal (1.248 pregões)
├── WIN$N_*_BRUTO.csv                ← WIN em M1, M5, M30, H1
├── WDO$N_*_BRUTO.csv                ← WDO nos mesmos timeframes
├── resultados_vol_0900-1100.csv     ← tabela da última rodada
└── resultados_vol_0900-1100_config.json  ← configuração que a gerou
```

Cada rodada grava o par `.csv` + `_config.json`, com janela, esquema de rótulo, número de
sementes, distribuição das classes e p-valor da permutação. É o que torna um resultado
rastreável meses depois.

---

## Ambiente

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy \
            tensorflow xgboost keras-tcn MetaTrader5 pytz
```

Python 3.13 no Windows. O pacote `MetaTrader5` só funciona no Windows e ainda não tem
versão para o Python 3.14 — se for baixar dados, use o mesmo interpretador do kernel do
notebook.

---

## Uma nota sobre o objetivo

Boa parte do que se publica sobre previsão de mercado com machine learning tem acurácia
alta e nenhum baseline, uma execução só e nenhum teste de significância, ou backtest sem
custo de transação. Esses trabalhos não replicam.

O que eu quero aqui é o contrário: um resultado que **eu mesmo consiga desmontar** e que
mesmo assim continue de pé. Se a conclusão for que não dá para prever, tudo bem, isso
também é informação, e é informação que quase ninguém publica.

### Isto aqui replica

Rodei o notebook inteiro duas vezes, do zero, e comparei número por número. Acurácias,
p-valores, resultado do backtest e do holdout: **todos idênticos**. Só os tempos de execução
mudaram.

Faço questão de registrar porque não é automático, e porque é justamente a parte que falta
nos trabalhos que eu critico no parágrafo acima. Fixar `np.random.seed` não basta quando há
TensorFlow envolvido: o framework tem gerador próprio, e sem prendê-lo as redes saem
diferentes a cada execução. Aqui cada arquitetura chama `tf.keras.utils.set_random_seed`
antes de ser construída, os modelos do scikit-learn recebem `random_state`, e a divisão dos
folds é cronológica, portanto determinística.

Para conferir, basta reiniciar o kernel e rodar tudo de novo. O detalhamento está na seção
5.9 do notebook.
