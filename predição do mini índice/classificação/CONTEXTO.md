# CONTEXTO DA SESSÃO

- **Última atualização:** 2026-08-17 00:40
- **Sessão nº:** 5
- **Status geral:** **ciclo principal FECHADO.** O notebook foi executado de ponta a ponta pelo
  utilizador, o **holdout foi aberto** (uma vez, conforme a regra) e o **resultado direcional é
  NULO em todas as frentes**. O que resta é a bateria de 5 variantes pré-registradas.

## 1. Objetivo da tarefa

Prever o comportamento do mini-índice (WIN) na janela das **11:00–11:59** a partir dos candles
das **09:00–10:59**, classificando em **Compra**, **Venda** ou **Lateral**. A operação pretendida
é entrar às 11:00 e sair às 12:00 (ou num alvo).

**Objetivo declarado pelo utilizador:** testar as arquiteturas, comparar resultados e **divulgar
como trabalho científico**. Isso eleva o padrão — ver [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md).

O notebook antigo [Predição do preço_M1.ipynb](Predição do preço_M1.ipynb) **será apagado**. Todo o
trabalho vivo está em
[classificação Compra-Lateral-Venda do WIN M5.ipynb](classificação%20Compra-Lateral-Venda%20do%20WIN%20M5.ipynb),
que **não o referencia** (verificado).

## 2. Já feito ✅

### Sessões 1–3 (resumo — detalhe em DOCS/)

- Auditoria completa do notebook antigo (115 células): 9 bugs concretos, ausência de baseline,
  razão amostra/feature de 1:8. Em [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md).
- **Dados recuperados.** O utilizador reexportou do MetaTrader após ajustar "Máx. barras no
  gráfico". Os quatro timeframes agora cobrem 2021-08 a 2026-08. Análise em
  [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md).
- **Teste de sinal definitivo com n = 1.240 — NULO.** Permutação com 200 embaralhamentos:
  **p = 0,333**.
- [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) com o protocolo de publicação e a **lista
  fechada de 5 variantes** (§7).
- Notebook M5 criado do zero.

### Sessão 4 (2026-08-14 a 16) — o grosso do trabalho

**Análises medidas (todas registradas no notebook):**

| Tema | Achado principal | Onde ficou |
|---|---|---|
| Comparação dos 4 timeframes | os 4 dão ganho negativo; M5 é o único viável para sequência | seção 2 |
| Correlação de features entre timeframes | `f_macd` tem correlação **−0,09** entre M1 e M5 — são features diferentes | seção 2 |
| Clusterização para rótulo (K-Means) | silhueta 0,23–0,30; rótulo instável no tempo (ARI 0,47) | seção 4 |
| DBSCAN | 93,4% dos dias num único cluster; instável ao `eps` | seção 4 |
| Abstenção por limiar de probabilidade | probabilidades **anti-informativas**; pior que sortear | seção 4 |
| Balanceamento por intercalação | **corrompe o baseline** — ganho triplica sem o modelo melhorar | seção 6 |
| EDA: informação mútua | 5 de 21 features passam de 2x a régua de ruído | seção 7 |
| EDA: comparação por classe | 3 de 21 a p<0,05; **nenhuma** sobrevive a Bonferroni | seção 7 |
| Seleção de features | **pior que sortear** o mesmo número (z = −0,88) | seção 15.1 |
| Arquitetura híbrida CNN-LSTM | ficou em 3º; "A Melhor" da teoria não se confirmou | seção 9 |
| Hiperparâmetros L2 e lr | medidos, não chutados | seção 10.7 |
| **Macro F1 vs P&L** | **correlação −0,536** — a métrica aponta contra o dinheiro | seção 10.10 |

**Mudanças de código no notebook:**

- Célula de conversão do CSV bruto do MetaTrader (seção 2.1), testada contra o arquivo oficial
- `f_tam_ordem` (Volume_Real / Volume_Tick) adicionada às features
- **5 features de contexto diário** (`c_gap`, `c_range_d1`, `c_atr_d1`, `c_pos_d1`, `c_dist_mm20`)
- Colunas ordenadas com **contexto primeiro**, com `COLUNAS_CONTEXTO` / `COLUNAS_JANELA`
- **Redes convertidas para duas entradas** (tensor sequencial + vetor de contexto)
- **Regularização L2 = 1e-3** nas camadas densas e convolucionais (antes não havia nenhuma).
  **Exceção medida:** a camada `TCN` do `keras-tcn` não aceita `kernel_regularizer`, então no
  TCN o L2 só alcança a cabeça e o ramo de contexto. Documentado na §9 e §10.5 do notebook.
- Bloco `DL_*` de hiperparâmetros nomeados, cada um justificado na seção 10
- `CUSTO_PONTOS` corrigido de **3,0 para 6,0** (o spread medido do WIN é 5 pontos)
- Backtest passou de in-sample para **out-of-fold**

### Sessão 5 (2026-08-16) — a execução completa e o holdout

**O utilizador rodou o notebook inteiro.** Todas as 20 células de código executaram sem erro.
Os números abaixo são da execução real, não de estimativa.

| Evidência | Resultado |
|---|---|
| Melhor modelo (XGBoost, 10 sementes) | acc 0,3604 — ganho **+0,85 p.p.** |
| Permutação (200 embaralhamentos) | z = 1,08, **p = 0,144** |
| Friedman (7 tabulares, 5 folds) | chi² = 5,65, **p = 0,463** |
| Nemenyi | CD = 4,03, amplitude 2,70 — **nenhum par difere** |
| Posição do baseline majoritário | **3º de 7** — bateu 4 dos 6 modelos reais |
| Kruskal nas 21 features | 3 a p<0,05; **0 sobrevivem a Bonferroni** |
| Backtest out-of-fold (custo 7 pts) | **−8.344 pontos (−R$ 1.668,80)** |
| Redes (3 sementes) | todas abaixo dos tabulares; **TCN em último (−0,0199)** |

**O HOLDOUT FOI ABERTO** — 244 dias, 2025-08-18 a 2026-08-12, uma única vez.

- Baseline 0,3238 | Modelo 0,3811 | **Ganho +5,7 p.p.** | +5.838 pts (R$ 1.167,60)
- **Parecia o melhor número do projeto.** Três testes o desmontaram:
  1. **Poder:** limiar mínimo detectável pré-registrado = 8,4 p.p.; observado 5,7 (z = 1,91,
     p bilateral 0,056) → abaixo do limiar
  2. **Deriva de classe:** a majoritária mudou de **Venda** (treino) para **Compra** (holdout),
     rebaixando o baseline. Contra a majoritária real: ganho **+3,7 p.p., p = 0,11**.
     **36% do ganho não era do modelo.**
  3. **P&L contra o acaso:** 20.000 estratégias aleatórias com as mesmas 161 operações →
     modelo no **percentil 78** (z = 0,79, p = 0,218). No walk-forward, o aleatório rendeu
     **+44,01 pts/operação** contra +36,26 do modelo no holdout.

> **A lição desta sessão:** o que impediu +5,7 p.p. e R$ 1.167 de virarem "descoberta" foi a
> tabela de decisão ter sido escrita **antes** de abrir. Está na §13.9 do notebook.

**Janela usada no holdout:** `09:00–10:59 → 11:00–11:59` (a de treino), **não** a
`09:00–11:59 → 12:00–12:59` que estava planejada. Decisão consciente e justificada na §13.5:
holdout serve para validar *o mesmo modelo* em dado novo; trocar a janela misturaria dois
efeitos. **Custo:** a hipótese das 11:30 (variante V3) perdeu o holdout limpo dela.

**Reorganização do notebook (sessão 5):**

- Só o título é `#`; toda seção é `##` numerada de 1 a 17 (antes "O que vem depois" e
  "Observações" não tinham número, e a §11 tinha subseções numeradas como 9.x)
- Tabela "Roteiro deste notebook" reescrita — listava 11 seções com a numeração errada
- §13 reescrita inteira em torno do holdout que aconteceu (9 subseções)
- Números da §15 sincronizados com a execução (permutação, Friedman, custo, baseline)
- §7.2 ganhou a demonstração de que o ranking de MI é ruído do estimador: `f_vol_real` e
  `f_atividade` correlacionam **0,942** e têm MI de 0,000 e 0,038

### Sessão 5, parte final — pair plots como diagnóstico de construção

Antes de aceitar pair plot no notebook, testei se ele acharia algo que a §7 não achou. Para
cada um dos **210 pares**, medi `MI(par 2D ; rótulo) − max(MI de cada uma sozinha)` contra
200 permutações do rótulo:

| | Real | Nulo (média) | Nulo p95 |
|---|---:|---:|---:|
| Maior ganho entre os 210 pares | 0,0333 | 0,0335 | 0,0373 |

**O melhor par fica abaixo da média do acaso (p = 0,485).** Não há estrutura bivariada. Isso
fecha uma lacuna do argumento da §15.1, que usava a possibilidade de interação para justificar
manter as 21 features — no nível de pares essa interação não existe.

Com isso, os pair plots entraram só como **diagnóstico de construção de feature**, em duas
células novas (a executar):

- **§4** — confirma o "V" que estava escrito como suspeita. `drawdown + runup = volatilidade`
  com erro de `3,5e-18`, posto 5 de 6. E o joelho: `r(retorno, drawdown)` é **−0,175** no ramo
  de alta e **−0,888** no de baixa; `r(retorno, runup)` é o espelho. O `pavio` da rotulagem
  correlaciona **+1,000** com o drawdown num ramo e **−0,244** no outro. A "estrutura" que o
  K-Means achou era a dobra das próprias fórmulas em retorno = 0.
- **§7.3** — formato dos 7 pares com `|r| > 0,8`: **só `f_vol_real` × `f_atividade` é reta
  apertada** (R² 0,887); os outros 6 são leques (R² 0,71–0,87), então cortar por correlação
  jogaria fora informação. E o par `f_dow_sin` × `f_dow_cos`: Pearson **+0,206** com
  dependência **perfeita** — raio 1,000000 ± 5e-17, 5 pontos distintos. Demonstração de que
  correlação baixa não é independência.

### Sessão 5, ajustes de redação

- **Vocabulário enxugado.** A seção "Para quem eu estou escrevendo" (glossário de mercado:
  spread, tick, pavio, RSI, MACD, ATR, engolfo, doji) foi **removida a pedido**. Ficou só
  `## O vocabulário deste projeto`, com 8 termos: janela de entrada, janela-alvo,
  Compra/Venda/Lateral, drawdown, **baseline, holdout, fold e walk-forward**. Os quatro
  últimos entraram porque estavam sendo usados na introdução sem definição, o `holdout`
  aparecia 16 células antes de ser explicado.
- **Travessões.** 340 de 470 trocados por vírgula. Os 139 que sobraram são os casos em que a
  vírgula embaralharia a leitura (trecho seguinte já com vírgula, travessões emparelhados) e
  os 4 marcadores de "não se aplica" em tabela. Faixas de horário (`–`, U+2013) e negativos
  (`−`, U+2212) não foram tocados, conferido.
- **Nota de correção removida.** Eu tinha deixado no texto um parágrafo explicando que a
  versão anterior dizia "maximiza a amostra" e estava errada. Como o trabalho ainda não foi
  divulgado, não há leitor a quem prestar contas, o texto simplesmente diz a coisa certa
  agora ("não custa amostra"). **O fato medido continua valendo:** `10:00–10:59 → 11:00–11:59`
  tem 1.221 dias contra 1.213 da escolhida, diferença de 0,7%, e a alternativa teria 12 velas
  em vez de 24.

### Sessão 5, re-execução completa (2026-08-17 00:04)

O notebook foi rodado inteiro de novo, do zero. Duas coisas a registrar.

**1. Ordem de execução corrigida.** Antes os `execution_count` estavam fora de ordem: o
registro rodou em 20 e o holdout em 21, então o `_config.json` gravado dizia
`"holdout_executado": false` quando ele tinha sido executado. Agora a sequência é **1 a 22,
monotônica, na ordem das células** — holdout em 21, registro em 22, e o JSON diz `true`.

**2. Reprodução bit a bit.** Comparei número por número contra a rodada anterior:

| Medição | Rodada 1 | Rodada 2 |
|---|---|---|
| XGBoost | 0,3604 ± 0,0061 | idêntico |
| Permutação | z = 1,08, p = 0,1443 | idêntico |
| Friedman | chi² = 5,651, p = 0,4634 | idêntico |
| Backtest out-of-fold | −8.344 pts | idêntico |
| MLP / CNN / LSTM / TCN | 0,3416 / 0,3549 / 0,3491 / 0,3342 | idêntico |
| Holdout | 0,3811, +5.838 pts | idêntico |

**Nada mudou na terceira casa decimal**; só os tempos de execução variaram. O ponto que
importa: **as redes reproduziram exatamente**, o que confirma que
`tf.keras.utils.set_random_seed(semente)` está controlando toda a aleatoriedade do TensorFlow.
Reprodução exata de pipeline com deep learning não é automática — é comum fixar só
`np.random.seed` e achar que resolveu. É uma afirmação forte para o paper e dá para verificar
rodando de novo.

### Sessão 5 — o holdout foi executado duas vezes, e por que isso não o contamina

**O fato:** o holdout foi aberto na primeira rodada e de novo na re-execução. A §13.3 do
notebook diz que abrir duas vezes destrói o holdout. Registro aqui o motivo de eu considerar
que este caso não se enquadra, para não ficar sem resposta se a pergunta surgir depois.

**O que a regra protege:** **reuso adaptativo** — usar o resultado do holdout para decidir
alguma coisa, mexer no modelo, e medir de novo até o número melhorar. Cada ciclo desses gasta
um pouco da independência do conjunto, e depois de alguns o holdout virou treino disfarçado.

**O que aconteceu aqui:** entre as duas aberturas mudaram apenas (a) texto — glossário,
travessões, explicações da tabela de janelas — e (b) duas células de diagnóstico de geometria
de feature (os pair plots), que não tocam em modelo, feature, rótulo nem janela. **Nenhuma
decisão foi informada pelo primeiro resultado.** Isso é **reprodução**, não segunda tentativa.

**A prova:** o número saiu idêntico (0,3811 / +5.838 pts), e os dois `_config.json` gravam a
configuração completa das duas rodadas. Se alguma coisa do lado do modelo tivesse mudado, o
número teria mudado junto.

> **O risco daqui em diante é real.** Se o notebook for re-executado depois de mexer em
> qualquer coisa do modelo, essa defesa some — passariam a ser duas medições diferentes no
> mesmo conjunto, e a segunda não valeria nada.
>
> **Ação recomendada:** voltar `EXECUTAR_HOLDOUT` para `False` agora que o número está
> registrado, e só ligar de novo se houver uma justificativa escrita antes.

## 3. Em andamento 🔧

Nada em execução. **As células de pair plot já foram executadas** (célula 14, ec=6; célula 25,
ec=12) e as duas figuras estão gravadas no notebook.

**Ponto de parada da sessão 4** (revisão do notebook antigo — mantido para referência):

1. **TCN** (§9): teoria da dilatação, bloco residual, campo receptivo. Ao conferir o código contra
   o texto, descobri que o `keras-tcn` **não aceita** `kernel_regularizer`. Medi as redes: o TCN
   tem **17.987 parâmetros** (o maior, 3,3× a CNN) e só **2 termos de L2** (o menor). Corrigido no
   código (`DL_DROPOUT_TCN`) e documentado na §9 e §10.5.
2. **LSTM / GRU / CNN-LSTM / Transformer** (§9): nenhuma entra como arquitetura nova — todas pioram
   a razão amostra/parâmetro. Medições que sustentam a decisão: GRU corta só **16%** dos parâmetros
   da LSTM no modelo completo (não 25%, porque contexto e cabeça não encolhem); Transformer no topo
   do espaço de busca usual chega a **459.395 parâmetros = 471 por amostra**, contra 5,6 da CNN.
3. **Vazamento do `early stopping`** (nova §5.6): achado metodológico mais transferível da revisão.
   Usar o mesmo conjunto em `validation_data=` e em `.predict()` faz o `restore_best_weights=True`
   devolver o máximo de uma curva em vez de uma estimativa. **O código deste notebook está correto**
   — usa `validation_split=0.2, shuffle=False` e nunca passa o fold de teste como validação.
   A antiga §5.6 virou §5.7 (não havia referências cruzadas a `§5.x`, então renumerar foi seguro).

4. **SVM / DBSCAN / SARIMA** (§4, §5.8, §8): as três seções mais bem executadas do notebook
   antigo — o SVM ajusta o scaler dentro do fold e otimiza F1-macro (o texto dele dizia
   acurácia, o código estava certo), e o DBSCAN já aplicava PCA antes de clusterizar.
   Nenhuma entra como método novo. O que entrou foram quatro medições:
   - **§8** — vetores de suporte são a contagem de parâmetros efetivos do SVM: com RBF,
     **98–100% dos pontos** viram vetor de suporte, mesmo regime de capacidade das redes.
   - **§4** — concentração de distâncias: o contraste `(dmáx−dmín)/dmín` cai de 29.925 (2
     dimensões) para **17** (120 dimensões), o que explica por que o `eps` do DBSCAN é
     incalibrável em alta dimensão e por que reduzir dimensão antes não é opcional.
   - **§8** — teto da família ARIMA/SARIMA nas 12.062 barras H1: ADF confirma `d=1`
     (preço p=0,883; retorno p≈0), e a ACF máxima dos retornos é **−0,029** → um AR(1)
     explicaria **0,086% da variância**. Previsão ingênua: RMSE 498 pts, 48,3% direcional.
     Vira linha de base univariada declarada para o paper.
   - **§5.8** — aviso metodológico: eliminar uma hipótese não confirma a alternativa (o
     notebook antigo usava o fracasso do SARIMA como "justificativa" para as 264 features).

O notebook está validado por smoke test (com TensorFlow real) e não tem pendência técnica.

**Estado do holdout: GASTO.** Foi aberto na sessão 5, uma vez, e não pode ser reaberto.
`EXECUTAR_HOLDOUT = True` na célula 36, com o resultado registrado nas saídas.

## 4. Próximos passos (planejado) 📋

1. **Re-executar a seção 9 com `N_SEMENTES_DL = 10`.** Os tabulares já rodaram com as 10
   sementes exigidas; as redes rodaram com 3. Isso não muda a conclusão (elas ficaram abaixo
   dos tabulares), mas **nenhum número de rede pode ir para o paper com 3 sementes**.
2. **Rodar a bateria de 5 variantes** ([DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) §7).
   Começar pela **V4 (prever volatilidade em vez de direção)** — é a de maior probabilidade a
   priori, e três achados independentes apontam para ela:
   - as únicas features que separam as classes são de volume/atividade (seção 7)
   - o contexto diário (medidas de volatilidade recente) bate as features da janela (seção 15.1)
   - a autocorrelação da volatilidade realizada é positiva; a da direção não é
3. **Testar ROCKET / MiniROCKET** (seção 9). É o método com melhor perfil para 969 amostras:
   kernels aleatórios não treinados + classificador linear.
4. **Migrar para scripts** com semente parametrizável, para o paper.
5. **Análise estatística formal** com correção de Holm-Bonferroni sobre todas as variantes.

> **Antes de qualquer coisa acima: voltar `EXECUTAR_HOLDOUT` para `False`** (célula 42). O
> número do holdout já está registrado nas saídas e no `_config.json`. Deixar a flag ligada
> significa que a próxima re-execução abre o holdout de novo — e se algo do modelo tiver
> mudado até lá, a medição fica contaminada sem aviso.

> **As variantes não têm holdout.** O de 244 dias foi gasto. Qualquer variante daqui em diante
> é avaliada em walk-forward, com o peso de evidência menor que isso implica — e isso **precisa
> constar no paper**. A alternativa seria reservar um segundo holdout dos dados mais recentes,
> mas isso reduziria ainda mais o treino de uma base que já é o teto da série disponível.

## 5. Decisões e raciocínio 🧠

### O resultado, e o que o sustenta

| Evidência | Resultado |
|---|---|
| Melhor modelo vs baseline (XGBoost, 10 sementes) | **+0,85 p.p.** |
| Teste de permutação (200 embaralhamentos) | **p = 0,144** |
| Friedman entre os 7 tabulares | **p = 0,463** — sem diferença detectável |
| Posição do baseline no Friedman | **3º de 7** |
| Kruskal nas 21 features | 0 sobrevivem a Bonferroni |
| Backtest out-of-fold com custo de 7 pts | **−8.344 pts (−R$ 1.668,80)** |
| **Holdout (244 dias, uma olhada)** | **+5,7 p.p. — abaixo do limiar de 8,4** |
| Holdout, P&L vs 20.000 aleatórios | **percentil 78** (p = 0,218) |
| Trocar timeframe (M1/M5/M30/H1) | os quatro negativos |
| Trocar janela horária (7 configurações) | máximo = o esperado do acaso |
| Selecionar features | pior que sortear |

### Alternativas já testadas e descartadas — não refazer

| Tentativa | Por que caiu |
|---|---|
| Achatar features por vela | 264 colunas para 1.220 amostras |
| Rótulo por K-Means / DBSCAN | sem estrutura de cluster; rótulo instável no tempo |
| Normalizar limiar pelo ATR **diário** | escala errada para janela de 1 hora → 82% "Lateral" |
| Abstenção por limiar de probabilidade | probabilidades anti-informativas |
| Cíclicas de mês e dia do mês | MI exatamente zero |
| Balanceamento por intercalação | corrompe o baseline |
| Seleção de features por MI ou Kruskal | pior que sortear |
| Arquitetura híbrida CNN-LSTM | 3º lugar; mais parâmetros, sem ganho |

### Armadilhas descobertas que valem para qualquer experimento futuro

1. **Baseline treinado em distribuição alterada** infla o ganho sem o modelo melhorar. Se um
   pré-processamento mexer na distribuição do treino, recalcule o baseline.
2. **`EarlyStopping(monitor="loss")` com `restore_best_weights=True`** restaura o ponto de
   **máximo overfitting**. Sempre `val_loss`.
3. **Macro F1 premia operar.** Correlação **−0,536** com o P&L. Nunca usar como alvo de
   otimização — seleciona pelos modelos que operam mais.
4. **RSI(14) em janela de 12 velas** é todo `NaN` e o `dropna()` apaga a base em silêncio.
5. **`|` dentro de crase numa tabela markdown** quebra a renderização.
6. **Ordenação de features entre folds é instável** — 13 features distintas em 25 escolhas.
7. **A classe majoritária pode mudar entre treino e holdout.** Quando muda, o
   `DummyClassifier` fica preso na do treino e o "ganho sobre o baseline" incorpora a deriva
   da distribuição. No holdout desta sessão isso valeu **2,0 dos 5,7 p.p. (36%)**. Em teste
   de período único, reportar sempre os dois baselines: o treinado e a majoritária do período.
8. **`keras-tcn` não aceita `kernel_regularizer`.** O TCN é a única arquitetura sem L2 nas
   convoluções, e a com mais parâmetros (17.987). Consistente com ela ter ficado em último.
9. **P&L de estratégia aleatória tem dispersão enorme.** No holdout, 20.000 sorteios deram
   desvio-padrão de **8.890 pontos** em 161 operações. Qualquer resultado econômico abaixo de
   ~2 desvios é ruído — e o `expectativa_pts_op` do aleatório no walk-forward (+44,01) já era
   maior que o do modelo.

### O que funcionou

- Recuperar o histórico (121 → 1.240 amostras)
- Resumir em vez de achatar (4,6:1 → 58:1)
- Features de contexto diário — o único bloco que bate as da janela
- O protocolo em si: impediu **cinco** falsos positivos de virarem descoberta

## 6. Estado do projeto / ambiente

**Raiz do projeto:** `C:\Users\marcos\Documents\GitHub\IA\predição do mini índice\classificação`
**Raiz git:** `C:\Users\marcos\Documents\GitHub\IA` (repo multi-projeto)

### Arquivos-chave

| Arquivo | Papel |
|---|---|
| [classificação Compra-Lateral-Venda do WIN M5.ipynb](classificação%20Compra-Lateral-Venda%20do%20WIN%20M5.ipynb) | **Notebook ativo.** 42 células, 20 de código, **17 seções**, todas executadas |
| [WIN$N_M5_BRUTO_COMPLETO.csv](WIN$N_M5_BRUTO_COMPLETO.csv) | **Base principal.** 138.400 barras, 1.240 amostras úteis |
| `WIN$N_{M1,M5,M30,H1}_2021...csv` | exportações brutas do MetaTrader, todas 2021-08 a 2026-08 |
| [coletar_m5.py](coletar_m5.py) | coleta via API do MT5 (alternativa à exportação manual) |
| [resultados_vol_0900-1100.csv](resultados_vol_0900-1100.csv) | tabela dos 11 modelos da rodada final |
| [resultados_vol_0900-1100_config.json](resultados_vol_0900-1100_config.json) | configuração que gerou a tabela |
| [README.md](README.md) | visão geral e roteiro — **atualizado na sessão 5** |
| [Predição do preço_M1.ipynb](Predição do preço_M1.ipynb) | notebook antigo, **a apagar** |

### Estrutura do notebook ativo

```
 1. Configuração               10. Configuração das redes (11 subseções)
 2. Carregamento e validação   11. Comparação estatística
 3. Engenharia de features     12. Avaliação econômica
 4. Rotulagem                  13. Holdout final (9 subseções)
 5. Protocolo (8 subseções)    14. Registro da rodada
 6. Tratamento dos dados       15. Conclusão
 7. Análise exploratória       16. O que vem depois
 8. Modelos tabulares          17. Observações
 9. Modelos sequenciais (DL)
```

Só o título do notebook é `#`. Toda seção é `##` numerada. As seções 10, 15, 16 e 17 são só
texto, sem célula de código.

**Configuração atual:** janela `09:00–10:59 → 11:00–11:59`, rótulo `vol`, 21 features
(16 janela + 5 contexto), 976 dias de trabalho + 244 de holdout (**gasto**), custo **7,0 pts**,
10 sementes nos tabulares e **3 nas redes** (subir para 10 antes do paper).

### Ambiente — atenção aos dois Pythons

| Uso | Interpretador |
|---|---|
| Scripts gerais (pandas, sklearn) | `python` do PATH — **Python 3.14** |
| **Qualquer coisa com TensorFlow ou MetaTrader5** | `C:\Users\marcos\anaconda3\python.exe` — **Python 3.13.9, TF 2.21** |

O Python 3.14 do PATH **não tem TensorFlow nem MetaTrader5**. Rodar o smoke test com ele faz a
seção 9 inteira ser pulada em silêncio — foi assim que um bug (`C_all` indefinido) passou
despercebido. **Sempre validar com o anaconda.**

`git` **não está no PATH** desta shell.

## 7. Bloqueios e pendências ⚠️

**Nenhum bloqueio técnico.** Dados recuperados, notebook validado, protocolo fechado.

**Decisão pendente do utilizador:** confirmar a reformulação da contribuição do paper —
benchmark metodológico ou resultado negativo ([DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) §2).
Com os números atuais, **resultado negativo** é o que os dados sustentam.

**Risco metodológico principal:** procurar variantes até uma "funcionar" é garimpagem de dados.
A defesa é a lista fechada de 5 variantes, que deve ser **commitada antes** de rodar, com todos os
resultados reportados e correção de Holm-Bonferroni.

**Dívida:** as redes rodaram com **3 sementes**, não 10. Os tabulares estão corretos. Antes de
qualquer número de rede entrar no paper, subir `N_SEMENTES_DL` para 10 e re-executar a seção 9.

**Consequência da sessão 5:** o holdout está gasto. As 5 variantes serão avaliadas só em
walk-forward, e isso reduz o peso de evidência delas. Precisa constar no paper.

**Inconsistência pequena a alinhar antes de escrever o paper:** o ganho do XGBoost aparece
como **+0,0085** na tabela de modelos (§8) e **+0,0111** no teste de permutação (§11.1). Os
dois estão corretos — a permutação reconstrói o walk-forward de forma um pouco diferente —
mas um revisor vai perguntar qual é o número. Decidir qual reportar e explicar a diferença.

**Flag ligada:** `EXECUTAR_HOLDOUT = True` na célula 42. Ver a nota no fim da seção 4.

## 8. Comandos úteis

```powershell
# Smoke test do notebook — SEMPRE com o anaconda (tem TensorFlow)
& "C:\Users\marcos\anaconda3\python.exe" -u "<scratchpad>\smoke_dl.py"

# Estrutura das seções
python -c "import json;nb=json.load(open('classificação Compra-Lateral-Venda do WIN M5.ipynb',encoding='utf-8'));import re;[print(m.group(0)) for c in nb['cells'] if c['cell_type']=='markdown' for m in re.finditer(r'^## .+$',''.join(c['source']),re.M)]"

# Procurar | não escapado em tabelas markdown (quebra a renderização)
# ver scratchpad/pipes.py

# Abrir o notebook
jupyter notebook "classificação Compra-Lateral-Venda do WIN M5.ipynb"

# Dependências
pip install pandas numpy scikit-learn matplotlib seaborn scipy tensorflow xgboost keras-tcn sktime
```

> Os scripts de análise da sessão 4 ficaram no scratchpad temporário e **não persistem**. Os
> resultados que importam estão todos registrados no notebook e em `DOCS/`.

## 9. Como retomar

Leia este arquivo, depois o notebook ativo — em especial a **seção 13 (Holdout)** e a
**seção 15 (Conclusão)**, que consolidam tudo com os números da execução real.

**O ciclo principal está fechado.** Não há mais nada a decidir sobre a configuração atual: o
resultado é nulo, o holdout foi aberto e confirmou o nulo, e tudo está registrado.

**Continue pelo passo 1 da seção 4:** subir `N_SEMENTES_DL` para 10 e re-executar a seção 9,
que é a única pendência que impede números irem para o paper. Depois, a variante **V4**.

**Seis avisos:**

1. **Valide com o anaconda, não com o Python do PATH.** O do PATH não tem TensorFlow e pula a
   seção 9 em silêncio.
2. **O holdout está GASTO.** Foi aberto na sessão 5. Não reabra, não reconfigure, não "teste
   só mais uma coisinha" nele. As variantes vivem em walk-forward.
3. **Não teste arquitetura nova** esperando que resolva. Onze modelos, quatro timeframes, sete
   janelas — tudo no ruído. O gargalo não é o modelo.
4. **Não corte features** olhando a tabela de informação mútua. Está medido: é pior que sortear.
5. **Não otimize por macro F1.** Correlação −0,536 com o P&L. Se houver busca de
   hiperparâmetros, o alvo é o resultado econômico out-of-fold.
6. **Desconfie de resultado econômico positivo.** O desvio-padrão do P&L aleatório é de ~8.900
   pontos em 161 operações. Antes de acreditar em qualquer lucro, rode o teste de randomização
   que está na §13.8.
