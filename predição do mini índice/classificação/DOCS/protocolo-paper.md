# Protocolo experimental para publicação científica

- **Data:** 2026-08-12
- **Objetivo do utilizador:** testar todas as arquiteturas do notebook, comparar resultados e
  divulgar como trabalho científico / paper.
- **Base factual:** [DOCS/analise-timeframes.md](analise-timeframes.md) e
  [DOCS/analise-notebook-M1.md](analise-notebook-M1.md).

---

## 1. O problema com o plano atual

Comparar 15 arquiteturas com **uma execução cada**, sobre **98 amostras**, com **conjunto de teste
de 19 dias**, não é publicável. Um revisor rejeita em três linhas:

1. **Sem repetições.** Cada arquitetura foi treinada uma vez, com uma semente. As diferenças
   observadas (0,05 a 0,50 de acurácia) são inteiramente compatíveis com variância de
   inicialização. Nenhuma comparação entre modelos é sustentável.
2. **Poder estatístico nulo.** n = 19 no teste ⇒ erro padrão ~±11 p.p. Não é possível distinguir
   nenhum par de modelos.
3. **Sem baseline.** Não há `DummyClassifier`, nem buy-and-hold, nem modelo linear. Sem isso não
   se sabe se algum resultado supera o trivial.
4. **Multiplicidade não corrigida.** 15 arquiteturas × busca de hiperparâmetros com Optuna sobre o
   mesmo conjunto de validação ⇒ inflação massiva da taxa de falsos positivos. O "melhor modelo"
   é, por construção, o mais sortudo.

**Isto não significa abandonar o projeto.** Significa que o mesmo trabalho, com protocolo correto,
é publicável — e que a contribuição precisa de ser reformulada.

---

## 2. Reformulação da contribuição

"Qual arquitetura é melhor" não é uma contribuição científica robusta neste domínio — a resposta
depende do ativo, do período e do label, e não generaliza. Duas alternativas defensáveis:

### Opção A — Benchmark metodológico rigoroso *(recomendada)*

> "Comparação sistemática de arquiteturas de aprendizagem profunda para classificação direcional
> intradiária do mini-índice Ibovespa, sob protocolo de validação walk-forward e avaliação
> económica."

A contribuição é o **protocolo e o benchmark reprodutível**, não o vencedor. Publicável mesmo que
nenhum modelo ganhe.

### Opção B — Resultado negativo *(honesta e sub-publicada)*

> "Arquiteturas profundas não superam baselines triviais na previsão direcional intradiária do
> mini-índice: evidência de 1.242 pregões."

Resultados negativos em finanças quantitativas são raros na literatura e valiosos — combatem o
viés de publicação. Dado o resultado preliminar (+3,2 p.p., 1,8 sigma, inconsistente entre folds),
**esta é provavelmente a conclusão a que se vai chegar**, e é uma conclusão publicável.

Decida a narrativa **depois** de correr os experimentos, nunca antes.

> **ATUALIZAÇÃO 2026-08-12 (sessão 3):** com a amostra completa (n = 1.240), o teste de sinal deu
> **nulo** — ganho de +0,7 p.p. sobre o baseline, p = 0,333 em teste de permutação com 200
> repetições. Ver [analise-timeframes.md §8](analise-timeframes.md). **A Opção B deixou de ser
> hipótese e passou a ser o resultado empírico.** Antes de a fixar como conclusão, correr a
> bateria de variantes pré-registada na §7 abaixo.

---

## 3. Protocolo mínimo obrigatório

### 3.1 Dados
- **Timeframe: M5.** Justificação medida em [analise-timeframes.md](analise-timeframes.md).
- **n ≈ 1.242 pregões** (fev/2021 – fev/2026), o teto da série disponível.
- Reportar exatamente: fonte (MetaTrader 5, `WIN$N`, rolagem por liquidez sem ajuste), período,
  critério de exclusão de dias, e o n final após limpeza.
- Declarar a **não-estacionariedade**: o índice variou +106% no período.

### 3.2 Repetições
- **≥ 10 sementes aleatórias por arquitetura** (30 é o ideal). Reportar **média ± desvio-padrão**,
  nunca um valor único.
- Fixar e publicar as sementes.

### 3.3 Validação
- **Walk-forward** (`TimeSeriesSplit`) com folds de tamanho reportado. Nunca embaralhar.
- **Nested CV**: a busca de hiperparâmetros ocorre *dentro* de cada fold de treino, nunca sobre o
  conjunto usado para reportar. Isto corrige o vazamento da célula 94.
- **Conjunto de teste final tocado uma única vez**, no fim.

### 3.4 Baselines obrigatórios
| Baseline | Porquê |
|---|---|
| Classe maioritária (`DummyClassifier`) | Piso absoluto |
| Aleatório estratificado | Piso para F1-macro |
| Regressão logística | Piso linear |
| ARIMA / SARIMA | Piso de séries temporais clássico (células 105–108 já têm) |
| Buy-and-hold | Piso económico |

### 3.5 Testes estatísticos
- **Comparação de múltiplos classificadores:** teste de **Friedman** + pós-hoc de **Nemenyi**, com
  diagrama de diferença crítica. Referência canónica: Demšar (2006), *JMLR*.
- **Comparação de previsões de séries temporais:** teste de **Diebold-Mariano**.
- **Correção para multiplicidade:** Holm-Bonferroni no mínimo; idealmente **White's Reality Check**
  ou **Hansen's SPA test**.
- **Teste de permutação:** embaralhar os labels ≥ 1.000× e situar o resultado real na distribuição
  nula. É o argumento mais convincente contra "isto é sorte".
- **Overfitting de backtest:** *Deflated Sharpe Ratio* / *Probability of Backtest Overfitting*
  (Bailey & López de Prado). Um revisor de finanças quantitativas vai pedir.

### 3.6 Avaliação económica
Acurácia não é resultado em finanças. Reportar obrigatoriamente:
- curva de equity com **custos de transação** (~2–3 pontos por operação no WIN, mais slippage);
- expectativa por trade, taxa de acerto, razão ganho/perda;
- Sharpe, Sortino, drawdown máximo, tempo em mercado;
- comparação com buy-and-hold no mesmo período.

### 3.7 Reprodutibilidade
- Código público (GitHub) com `requirements.txt` e versões fixas.
- Sementes, hiperparâmetros finais e splits publicados.
- Dados: se não puderem ser redistribuídos, publicar o script de recolha e os hashes.

---

## 4. Arquiteturas a comparar

Manter a lista do notebook, mas organizada em famílias e com os baselines a par:

| Família | Modelos |
|---|---|
| Trivial | Maioritária, aleatório estratificado |
| Linear | Regressão logística, SVM linear |
| Clássico de séries temporais | ARIMA/SARIMA |
| Ensembles de árvores | Random Forest, GradientBoosting, XGBoost |
| Kernel | SVM (RBF, poly) |
| Redes densas | MLP |
| Convolucionais | CNN 1D, CNN multi-entrada |
| Recorrentes | LSTM, GRU, BiLSTM |
| Convolucionais dilatadas | TCN |
| Atenção *(opcional)* | Transformer pequeno |

**Nota realista:** com ~1.242 amostras, as famílias profundas estão fora do seu regime de dados.
Isso é parte do achado, não um defeito do desenho — mas tem de ser dito explicitamente no paper,
não escondido.

---

## 5. Estrutura sugerida do paper

1. **Introdução** — problema, contexto do mini-índice, lacuna na literatura (pouca evidência
   sobre o mercado brasileiro intradiário com protocolo rigoroso).
2. **Trabalhos relacionados** — previsão direcional com deep learning; crítica metodológica
   (López de Prado; Demšar; Arnott, Harvey & Markowitz "A Backtesting Protocol").
3. **Dados** — MT5, `WIN$N`, M5, 2021–2026, limpeza, não-estacionariedade.
4. **Definição do problema e rotulagem** — janela 09:00–10:59 → 11:00–11:59; regra de label;
   análise de sensibilidade aos limiares (**seção crítica** — mostrar que o resultado não depende
   de escolher 50/250).
5. **Metodologia** — features, arquiteturas, protocolo walk-forward, nested CV, sementes.
6. **Resultados** — tabela média ± dp por arquitetura, diagrama de diferença crítica, teste de
   permutação, avaliação económica.
7. **Discussão** — porque os resultados são o que são; regime de dados pequenos; limitações.
8. **Conclusão** — honesta.

### Referências que o revisor vai esperar
- Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR.
- López de Prado (2018), *Advances in Financial Machine Learning* — caps. 3 (triple-barrier),
  7 (purged CV), 11–12 (backtest overfitting).
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
- Arnott, Harvey & Markowitz (2019), *A Backtesting Protocol in the Era of Machine Learning*.
- Fischer & Krauss (2018), *Deep learning with LSTM networks for financial market predictions*.

### Veneus realistas
- **Nacionais:** BRACIS, ENIAC, KDMiLe, SBSI, Revista Brasileira de Finanças.
- **Internacionais:** *Expert Systems with Applications*, *Neural Computing and Applications*,
  *Journal of Forecasting*, *Quantitative Finance*.
- **Pré-print:** arXiv (q-fin.ST ou cs.LG) — recomendado antes da submissão.

---

## 6. Ordem de execução recomendada

1. Recuperar o M5 completo (2 blocos) ⇒ ~1.242 amostras.
2. Corrigir os bugs listados em [analise-notebook-M1.md](analise-notebook-M1.md) — em especial o
   vazamento da célula 94 e o `EarlyStopping(monitor='loss')`. **Resultados produzidos com esses
   bugs não podem entrar no paper.**
3. Reescrever o pipeline como **scripts** (não notebook) com semente parametrizável e log
   estruturado dos resultados. Um notebook de 40 MB não é reprodutível.
4. Correr o benchmark completo: 10 arquiteturas × 10 sementes × 5 folds = 500 treinos.
   Orçamentar tempo de computação.
5. Análise estatística (Friedman + Nemenyi + permutação).
6. Avaliação económica.
7. Escrever, começando pela seção de Metodologia.

**Regra de ouro:** decidir os critérios de sucesso e os testes estatísticos **antes** de olhar
para os resultados finais. Idealmente, registar o plano (pre-registration) num repositório com
data. É a defesa mais forte contra a acusação de garimpagem de dados.

---

## 7. Bateria de variantes pré-registada (2026-08-12)

Depois do resultado nulo com n = 1.240, a tentação é procurar variantes até uma "funcionar". Isso
é garimpagem de dados e destrói a validade do paper. **A defesa é fixar a lista agora, testar
tudo, e reportar tudo — inclusive o que falhar.**

Lista fechada de 5 variantes. Nenhuma outra entra sem ser declarada como exploratória.

| # | Variante | Hipótese que testa | Custo |
|---|---|---|---|
| V1 | Adicionar features do WDO à janela 09:00–10:59 | Falta informação exógena (o WDO já está no disco) | Baixo |
| V2 | Adicionar futuro do S&P (ES) na mesma janela | O driver das 11h é externo e não está no input | Médio (obter dados) |
| V3 | Mover a janela-alvo para 11:30–12:30 | O evento previsível é a abertura americana, não as 11h | Baixo |
| V4 | Prever **volatilidade** em vez de direção | Volatilidade é autocorrelacionada; direção não é | Baixo |
| V5 | Regressão do retorno em vez de classificação | A discretização em 3 classes destrói informação | Baixo |

**V4 é a que tem maior probabilidade a priori de dar resultado positivo** — a persistência da
volatilidade é um dos factos estilizados mais robustos em finanças, e o próprio notebook já a
propõe (markdown "Alternativa 3"). Um paper que reporte *"direção: nula; volatilidade: sinal"* é
substancialmente mais forte do que só o resultado negativo.

### Regras de execução

- Correr **todas as 5**, mesmo depois de uma dar positivo.
- Reportar as 5 no paper, com o mesmo protocolo (walk-forward, baseline, permutação).
- **Corrigir para multiplicidade:** com 5 variantes testadas, aplicar Holm-Bonferroni. Um p = 0,04
  isolado numa das cinco **não** é significativo depois da correção.
- Registar esta lista com data (commit no git) **antes** de correr — é a prova de
  pre-registration.
