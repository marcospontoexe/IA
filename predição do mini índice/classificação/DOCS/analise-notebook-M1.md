# Análise técnica — Predição do preço_M1.ipynb

- **Data da análise:** 2026-08-12
- **Ficheiro analisado:** [Predição do preço_M1.ipynb](../Predição do preço_M1.ipynb) (115 células, ~5.800 linhas de código)
- **Método:** leitura estática de todas as células, inspeção dos outputs guardados e verificação
  factual dos ficheiros de dados no disco. Nenhuma célula foi executada; nenhum ficheiro do
  projeto foi modificado.

---

## Resumo

A engenharia está boa e a documentação é excelente, mas o projeto está estrangulado por **falta de
dados**: 98–103 dias de amostra contra 432–854 features. Nenhum dos ~15 modelos testados poderia
funcionar, e os resultados confirmam-no.

---

## 1. O que está correto e deve ser preservado

- **Documentação conceitual.** As células markdown sobre features vs. labels, data leakage,
  drawdown/run-up e escolha do ticker (`WIN$N`, rolagem por liquidez sem ajuste) estão corretas e
  bem fundamentadas.
- **Divisão cronológica.** `train_test_split(..., shuffle=False)` e `TimeSeriesSplit` em todo o
  notebook. Este é o erro nº 1 em projetos de trading e foi evitado.
- **Features de contexto diário sem vazamento.** Verificado linha a linha:
  `atr_dia_anterior`, `mm20_diaria`, `range_dia_anterior` e `posicao_range_d1` usam `.shift(1)`
  corretamente. O `gap` compara a abertura do dia (09:00) com o fechamento do dia anterior — ambos
  conhecidos antes das 11:00. Não há fuga do futuro nestas features.
- **Células 113 e 114** (abordagem enxuta) — de longe a melhor parte do notebook. Ver seção 6.

---

## 2. O problema fatal: proporção amostra/feature

| Dataset | Dias | Features | Razão |
|---|---|---|---|
| [dataset_final.csv](../dataset_final.csv) | 103 | 851 | **1 : 8,3** |
| [dataset_hibrido_win_wdo.csv](../dataset_hibrido_win_wdo.csv) | 98 | 429 | **1 : 4,4** |

O saudável é o inverso: **10:1 a 50:1 a favor das amostras**. Está a pedir-se a uma rede de
~300 mil parâmetros que aprenda a partir de 78 exemplos de treino.

O output da célula 45 mostra exatamente o que a teoria prevê:

```
accuracy: 1.0000 - loss: 0.0333  |  val_accuracy: 0.2667 - val_loss: 1.7581
```

100% no treino, 26% na validação. Memorização pura.

### Resultados de todos os modelos no conjunto de teste

| Célula | Modelo | Acurácia teste | F1-macro |
|---|---|---|---|
| 45 | MLP | 0,25 | 0,15 |
| 49 | MLP + Optuna | 0,26 | 0,19 |
| 54 | CNN multi-input | 0,25 | 0,24 |
| 56 | CNN + Optuna + CV | 0,42 | 0,35 |
| 62 | TCN | 0,25 | 0,25 |
| 66 | LSTM (fixa) | 0,50 | 0,36 |
| 71 | LSTM + Optuna | 0,26 | 0,20 |
| 73 | LSTM + Optuna (simples) | 0,26 | 0,27 |
| 82 | (variante) | 0,05 | 0,04 |
| 86 | (variante) | 0,10 | 0,09 |
| 92 | XGBoost-DART | 0,21 | 0,12 |
| 94 | XGBoost + Optuna | 0,32 | 0,24 |
| 96 | SVM + Optuna | 0,16 | 0,11 |

**O conjunto de teste tem 19–20 dias.** Com n = 20, o erro padrão da acurácia é ~±11 pontos
percentuais. Estatisticamente, **0,16 e 0,50 são o mesmo número**. O LSTM da célula 66 com 50% não
é "o melhor modelo" — é a cauda de uma distribuição aleatória. Selecioná-lo seria fazer escolha
sobre ruído.

### Causa raiz da escassez de dados

[WIN$N_M1_BRUTO.csv](../WIN$N_M1_BRUTO.csv) tem **exatamente 100.000 linhas** e cobre de
`2025-06-06` a `2026-02-20` — 8 meses. Isso é o teto de exportação da interface do MetaTrader, não
o limite do histórico disponível.

**Solução:** a célula 6 já usa `mt5.copy_rates_range()`, que não tem esse teto. Correr em blocos
anuais de 2015 a 2026 em M5 dá ~2.700 dias úteis. Isto sozinho muda o projeto de "impossível" para
"talvez".

---

## 3. Segundo problema: a definição do label

```python
condicoes = [
    (retorno_total > 50)  & (pavio_contra < 250),   # Compra
    (retorno_total < -50) & (pavio_contra < 250),   # Venda
]
label = np.select(condicoes, labels, default='Lateral')
```

1. **"Lateral" é uma classe-lixo.** Junta o mercado parado, o movimento forte mas ruidoso (subiu
   300 pts com 300 de drawdown) e as reversões — três fenómenos diferentes com o mesmo rótulo, e
   ~1/3 dos dados.
2. **Limiares absolutos numa série não-estacionária.** 50/250 pontos com o índice a 137.000
   (jun/2025) não significam o mesmo com ele a 193.000 (fev/2026) — 40% de diferença de escala.
   Normalizar por ATR: `retorno > 0,3 × ATR_do_dia`.
3. **Isto é o triple-barrier method feito à mão** (López de Prado, *Advances in Financial ML*,
   cap. 3). Vale usar a formulação canónica.

---

## 4. Terceiro problema: a janela de entrada não contém a causa

O próprio markdown do notebook diz que o driver das 11h é a abertura americana às 11:30. Mas as
features param às 10:59 e a única informação exógena é o WDO. Faltam:

- futuro do S&P (ES) na janela 09:00–10:59;
- movimento overnight (Ásia/Europa);
- calendário macro (payroll, CPI, Copom, FOMC).

Está a pedir-se ao modelo que preveja o efeito de informação que ele nunca vê.

---

## 5. Bugs concretos

| Célula | Problema |
|---|---|
| **37** | `try:` com `df = pd.read_csv(...)` indentado a 8 espaços e o resto do bloco a 4 ⇒ **IndentationError**. A célula não corre. |
| **56, 60, 73** | `NUM_TIMESTEPS = 240`, mas o `dataset_hibrido_win_wdo.csv` no disco foi gerado com `quantidade_velas = 60` (429 features = 9 contexto + 7×60) ⇒ **KeyError** em `_c61`. Os outputs guardados são de outra versão do ficheiro — não são reproduzíveis. |
| **94** | `evals=[(dtest_final, 'test')]` + `early_stopping_rounds=50` ⇒ o número de árvores é escolhido a olhar para o conjunto de teste. **Vazamento direto.** |
| **51, 56, 60, 62, 71** | Treino final com `EarlyStopping(monitor='loss', restore_best_weights=True)` — monitorizar a perda de *treino* garante restaurar o ponto de máximo overfitting. |
| **108** | `s#erie_precos = df['vol'].dropna()` ⇒ avalia `s`, **NameError**. |
| **62** | `TimeSeriesSplit(n_splits=20)` sobre ~78 amostras ⇒ folds de validação com ~3 dias. F1-macro sobre 3 amostras não é métrica. |
| **21, 23** | `df_completo.replace(0, 1e-9, inplace=True)` aplica a *todas* as colunas; um `Volume_Tick` zerado faz explodir `tamanho_medio_ordem = Volume_Real / Volume_Tick`. |
| **37, 45, 49** | Balanceamento por `iloc[:min_samples]` descarta os dias **mais recentes** das classes maioritárias e intercala C/V/L de épocas diferentes. Substituir por `class_weight` no `fit()`. |
| geral | `exit()` dentro de notebook mata o kernel. Usar `raise` ou um bloco `if`. |

**Inconsistência de escopo:** o título e a célula 21 tratam do intervalo 11:00–11:59, mas a célula
23 (que gera o `dataset_hibrido`, o mais usado) usa features 09:00–09:59 e label 10:00–10:59.
Precisa de ser confirmado qual é o alvo oficial.

---

## 6. Duas coisas que faltam completamente

**Baseline.** Até à célula 114 não existe nenhum `DummyClassifier`. Foram comparados 15 modelos
entre si, mas nenhum contra "chutar sempre a classe maioritária". Com a distribuição 52/61/62, o
baseline é ~35% — ou seja, **quase todos os modelos perdem para chutar sempre "Venda"**.

**Avaliação económica.** Acurácia não é P&L. Com ~2–3 pontos de custo por operação e alvo de 50
pontos, um modelo com 40% de acerto em 3 classes ainda pode perder dinheiro. Falta curva de equity,
expectativa por trade, drawdown máximo — e um **teste de permutação** (embaralhar os labels 1.000×
e ver onde o resultado real cai na distribuição). É isto que separa sinal de sorte.

---

## 7. Plano recomendado

A célula 114 já tem a lógica certa, incluindo o critério de paragem honesto:

```python
print(f">>> Ganho de acuracia do GB sobre o baseline: {ganho:+.3f}  "
      + ("(ha indicio de sinal)" if ganho > 0.03 else "(SEM sinal util: nao supera o acaso)"))
```

Nunca foi executada (não tem outputs). **Correr antes de mais nada.**

1. **Recuperar histórico** via `mt5.copy_rates_range` em blocos ⇒ meta de ≥ 1.000 dias úteis (M5).
   Sem isto, nada mais importa.
2. **Manter a abordagem enxuta** da célula 113: ~15 features agregadas por dia, não 429 achatadas.
   Com 1.000 dias dá 66:1 — saudável.
3. **Correr a célula 114** (walk-forward + baseline). Se o GB não bater o Dummy em ≥ 3 dos 5 folds,
   **parar e reformular o label** — não testar outra arquitetura. Trocar MLP por LSTM por TCN sobre
   um label mau foi o que consumiu a maior parte deste notebook.
4. **Redefinir o label**: limiares em ATR, e considerar 2 classes + abstenção (probabilidade abaixo
   do limiar = não opera) em vez de forçar "Lateral".
5. **Só então** voltar a redes neurais — e só se um GradientBoosting simples já mostrar sinal.

### Sobre a escolha de arquitetura

Com dados diários agregados, MLP/CNN/LSTM/TCN vão ter desempenho praticamente igual. LSTM e TCN só
ganham quando há muitas sequências longas; com ~1.000 amostras, **gradient boosting sobre features
agregadas é a escolha certa** e poupa semanas.

### Alternativas já descartadas (não repetir a análise)

- *Trocar de arquitetura de rede* — 15 variantes testadas, todas no ruído.
- *Optuna com mais trials* — com validação de 19 dias, otimiza o ruído; mais trials pioram
  (overfitting na seleção de hiperparâmetros).
- *Labels por clusterização* (K-Means na célula 15, DBSCAN na 17) — abandonado a favor da regra
  explícita da célula 19.
- *Manter os datasets achatados* — origem do problema.

---

## 8. Dívida técnica

O notebook tem **40 MB** por causa dos outputs embutidos, o que o torna inviável em git. Vale mover
as ~15 células de modelos descartados para `DOCS/modelos_testados.md` e limpar os outputs.
