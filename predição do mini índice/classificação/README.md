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

### Status: resultado nulo até agora

Não vou enfeitar. Onze modelos testados, nenhum bateu o baseline trivial:

| Modelo | Acurácia | Ganho sobre o baseline |
|---|---|---|
| **Baseline (classe majoritária)** | **0,352** | — |
| MLP | 0,354 | −0,000 |
| LSTM | 0,345 | −0,010 |
| XGBoost | 0,343 | −0,009 |
| CNN 1D | 0,343 | −0,011 |
| Gradient Boosting | 0,340 | −0,012 |
| Random Forest | 0,340 | −0,012 |
| TCN | 0,335 | −0,019 |
| SVM (RBF) | 0,327 | −0,025 |
| Regressão logística | 0,305 | −0,047 |

Teste de permutação com 200 embaralhamentos: **p = 0,33**. Friedman entre os modelos:
**p = 0,079** — nem diferença entre eles dá para detectar. E no backtest *out-of-fold*, já
com custo de 3 pontos por operação, a coisa fecha em **−13.115 pontos (−R$ 2.623)**.

Ou seja: prever direção nessa janela, com essas features, não funciona. É um resultado, e
é por isso que ele está aqui em vez de estar escondido.

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
**Situação:** resultado nulo. Antes de fechar essa conclusão, quero rodar as variantes de
adicionar o dólar, adicionar o futuro do S&P, e mover a janela-alvo para depois das 11:30.

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
├── CONTEXTO.md                      ← onde o trabalho parou
├── CLAUDE.md                        ← índice da documentação
└── DOCS/
    ├── analise-timeframes.md        ← por que M5, escolha da janela, testes de sinal
    └── protocolo-paper.md           ← protocolo para publicação científica
```

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
mesmo assim continue de pé. Se a conclusão for que não dá para prever, tudo bem — isso
também é informação, e é informação que quase ninguém publica.
