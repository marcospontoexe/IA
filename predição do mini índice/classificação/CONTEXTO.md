# CONTEXTO DA SESSÃO

- **Última atualização:** 2026-08-12 21:40
- **Sessão nº:** 3
- **Status geral:** em andamento — dados resolvidos (n = 1.240); **teste de sinal deu NULO**;
  aguarda decisão sobre a bateria de variantes pré-registada

## 1. Objetivo da tarefa

Construir um classificador que, a partir dos candles do mini-índice (WIN) na janela das
09:00–10:59, preveja o comportamento do mercado na janela das 11:00–11:59, rotulado em três
classes: **Compra**, **Venda** e **Lateral**. A intenção final é executar uma ordem às 11:00 e
encerrá-la às 12:00.

**Objetivo declarado na sessão 2:** testar todas as arquiteturas do notebook, comparar os
resultados e **divulgar o trabalho como paper científico**. Isto eleva o padrão exigido — ver
[DOCS/protocolo-paper.md](DOCS/protocolo-paper.md).

## 2. Já feito ✅

- **Leitura e auditoria completa** do notebook [Predição do preço_M1.ipynb](Predição do preço_M1.ipynb)
  (115 células, ~5.800 linhas de código). Nenhuma célula foi executada nem modificada.
- **Diagnóstico entregue ao utilizador** (só no chat até este checkpoint; agora persistido em
  [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md)), cobrindo:
  - proporção amostra/feature inviável;
  - causa raiz da escassez de dados (teto de exportação do MetaTrader);
  - problemas na definição do label;
  - ausência de features exógenas que expliquem o alvo;
  - 9 bugs concretos, com número de célula;
  - ausência de baseline e de avaliação económica;
  - plano de correção em 5 passos priorizados.
- **Verificação factual dos ficheiros de dados** no disco (datas de cobertura, número de linhas,
  número de colunas dos datasets) — os números da análise foram medidos, não estimados.
- **Criação dos ficheiros de contexto**: este `CONTEXTO.md`, o [CLAUDE.md](CLAUDE.md) do projeto
  (índice) e [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md) (análise detalhada).

### Sessão 2 (2026-08-12)

- **Análise medida dos 4 ficheiros brutos do WIN** (M1, M5, M30, H1) — cobertura, dias úteis,
  qualidade, histórico recuperável. Registada em
  [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md). **Achado principal: o M1 é o pior dos
  quatro ficheiros (121 amostras úteis) e o M5, já no disco, tem 891.**
- **Teste de sinal preliminar executado** com M5 + lógica das células 113/114: ganho de +3,2 p.p.
  sobre o baseline, ~1,8 sigma, inconsistente entre folds. Indício fraco, não conclusivo.
- **Protocolo experimental para publicação** redigido em
  [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md).
- [CLAUDE.md](CLAUDE.md) atualizado com o índice dos novos documentos e 2 regras novas.

### Sessão 3 (2026-08-12)

- **Histórico M5 recuperado pelo utilizador** via MetaTrader (ajuste de "Máx. barras no gráfico").
  Ficheiro `WIN$N_M5_202108120900_202608121830.csv` convertido para
  [WIN$N_M5_BRUTO_COMPLETO.csv](WIN$N_M5_BRUTO_COMPLETO.csv):
  **1.240 amostras úteis**, 2021-08-12 a 2026-08-12, 138.400 barras.
  Fuso validado: 100,00% de coincidência de preços em 99.999 timestamps comuns.
  **1.248 dias distintos = o mesmo número do M30 e do H1 ⇒ teto do histórico confirmado.**
- **Teste de sinal definitivo executado com n = 1.240 — RESULTADO NULO.**
  Ganho sobre o baseline: +0,007 (label fixo) e −0,007 (label normalizado por volatilidade).
  **Teste de permutação (200 repetições): z = 0,44, p = 0,333 — não significativo.**
  Detalhe completo em [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md) §8.
- **Bateria de 5 variantes pré-registada** em [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) §7.
- **Notebook novo criado e testado**:
  [classificação Compra-Lateral-Venda do WIN M5.ipynb](classificação%20Compra-Lateral-Venda%20do%20WIN%20M5.ipynb)
  — 29 células, reorganiza a metodologia do notebook M1 para M5 com protocolo de publicação.
  Validado por smoke test: todas as células não-DL correm sem erro.
- **Sweep de 7 janelas horárias** (só nos primeiros 80% dos dias; holdout preservado) —
  nenhuma janela é demonstravelmente melhor. Registado em
  [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md) §9.
- **Erro de implementação encontrado e corrigido:** `RSI(14)` em janelas de 12 candles (1h em M5)
  é todo `NaN` e apagava silenciosamente todas as amostras. Ver §10 do mesmo documento.

## 3. Em andamento 🔧

Nada em execução. Nenhuma célula do notebook foi editada em nenhuma das três sessões.

**Ponto exato de parada:** o notebook M5 está criado, testado e pronto a correr. O resultado
continua nulo em todos os testes feitos até agora — incluindo a avaliação económica out-of-fold,
que dá **−13.115 pontos (−R$ 2.623)** depois de custos.

**Estado do holdout:** os 244 dias finais (2025-08-18 a 2026-08-12) **nunca foram tocados**. A
flag `EXECUTAR_HOLDOUT` no notebook está a `False` de propósito. O holdout aceita **uma única**
configuração — não o abra antes de fechar todas as decisões.

**Próximo passo imediato:** correr **V4 (previsão de volatilidade em vez de direção)** — é a
variante com maior probabilidade a priori de dar positivo e o custo é baixo. Mas ver a regra de
execução: as 5 variantes devem ser corridas e reportadas, não só a que funcionar.

## 4. Próximos passos (planejado) 📋

Ordem recomendada — **não pular o passo 1**, os restantes dependem dele.

1. **Recuperar o histórico M5 completo.** Meta: **~1.242 amostras úteis** (teto da série; a
   corretora tem desde ~2021-02). Duas vias:
   - **Via A (rápida):** MetaTrader → Ferramentas → Opções → Gráficos → "Máx. barras no gráfico"
     = **Ilimitado** → reiniciar o MT5 → abrir gráfico `WIN$N` M5 → `Home` até parar de carregar →
     exportar. O corte atual é por contagem de barras, não por data: o
     [WIN$N_M5_BRUTO.csv](WIN$N_M5_BRUTO.csv) começa a meio do pregão (`2022-06-10 13:35`), a
     assinatura clássica de um limite `maxbars` de 100.000.
   - **Via B (reproduzível, preferível para o paper):** executar
     [coletar_m5.py](coletar_m5.py) — descarrega por blocos anuais via `mt5.copy_rates_range`,
     valida o fuso contra o ficheiro antigo e imprime o diagnóstico de amostras úteis.
     **Requer o interpretador do kernel do notebook (Python 3.13.9), não o do PATH** (o Python
     3.14 do sistema não tem o pacote `MetaTrader5` e ainda não tem wheel disponível).

   *Enquanto isso não estiver feito, já se pode trabalhar com as 891 amostras do M5 atual.*
2. **Manter a abordagem enxuta.** Usar a lógica da célula 113 (~15 features agregadas por dia)
   como gerador oficial de `X`/`y`, e **abandonar** os datasets achatados de 429–851 colunas.
   Com 1.242 dias e 15 features a razão fica ~78:1, saudável. Para arquiteturas sequenciais
   (CNN/LSTM/TCN) usar o tensor 3D `(1242, 24, 7)` derivado do M5.
3. **Executar a célula 114** (walk-forward de 5 folds, GradientBoosting vs `DummyClassifier`).
   Critério de parada já codificado na própria célula: ganho de acurácia sobre o baseline > 0,03.
   **Se o GB não bater o baseline em ≥ 3 dos 5 folds, parar e voltar ao passo 4 — não testar
   outra arquitetura.**
4. **Redefinir o label** (só se o passo 3 falhar, ou em paralelo como variante):
   - trocar limiares absolutos (50 / 250 pontos) por múltiplos da **volatilidade da própria janela
     11:00–11:59** — **não** pelo ATR diário: isso já foi testado na sessão 2 e produziu 82% de
     "Lateral", destruindo o sinal (ver [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md), §5);
   - considerar 2 classes + abstenção (probabilidade abaixo de um limiar ⇒ não opera), em vez de
     forçar "Lateral" como terceira classe;
   - avaliar o *triple-barrier method* (López de Prado, *Advances in Financial ML*, cap. 3), que é
     a formulação canónica do que já se está a fazer à mão.
5. **Adicionar features exógenas** à janela 09:00–10:59: futuro do S&P (ES), movimento overnight
   (Ásia/Europa) e flag de calendário macro (payroll, CPI, Copom, FOMC). O WDO já está incluído.
6. **Avaliação económica** (só depois de haver sinal estatístico): curva de equity com custos
   (~2–3 pontos por operação), expectativa por trade, drawdown máximo, e **teste de permutação**
   (embaralhar os labels 1.000× e verificar onde o resultado real cai na distribuição).
7. **Corrigir os bugs** listados na secção 7 e em [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md).
   **Obrigatório antes de qualquer resultado destinado ao paper** — sobretudo o vazamento da
   célula 94 e o `EarlyStopping(monitor='loss')`.
8. **Higiene do notebook:** mover as ~15 células de modelos descartados para
   `DOCS/modelos_testados.md` e limpar os outputs (o ficheiro tem 40 MB, inviável em git).
9. **Migrar de notebook para scripts** com semente parametrizável e log estruturado, para poder
   correr 10 arquiteturas × 10 sementes × 5 folds. Requisito do paper.
10. **Análise estatística formal** (Friedman + Nemenyi + teste de permutação) e **avaliação
    económica** com custos. Ver [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md).

## 5. Decisões e raciocínio 🧠

**Diagnóstico central — o projeto está limitado por dados, não por arquitetura.**

| Dataset | Dias | Features | Razão |
|---|---|---|---|
| [dataset_final.csv](dataset_final.csv) | 103 | 851 | 1 : 8,3 |
| [dataset_hibrido_win_wdo.csv](dataset_hibrido_win_wdo.csv) | 98 | 429 | 1 : 4,4 |

O saudável é o inverso (10:1 a 50:1 a favor das amostras). A célula 45 mostra o sintoma clássico:
`accuracy: 1.0000 / val_accuracy: 0.2667`. Memorização pura.

**Todos os ~15 modelos já testados ficaram no ruído** (acurácia de teste entre 0,05 e 0,50; MLP,
MLP+Optuna, CNN, TCN, LSTM, LSTM+Optuna, XGBoost, XGBoost+Optuna, SVM). O conjunto de teste tem
19–20 dias ⇒ erro padrão ~±11 p.p. **0,16 e 0,50 são estatisticamente o mesmo número.** O LSTM que
deu 50% não é "o melhor modelo", é a cauda de uma distribuição aleatória; escolhê-lo seria seleção
sobre ruído. Por isso a recomendação é não continuar a trocar de arquitetura.

**Causa raiz da escassez de dados:** [WIN$N_M1_BRUTO.csv](WIN$N_M1_BRUTO.csv) tem **exatamente
100.000 linhas** e cobre `2025-06-06` a `2026-02-20`. Isso é o teto de exportação da interface do
MetaTrader, não o limite do histórico disponível. A API `mt5.copy_rates_range()` (já usada na
célula 6) não tem esse teto.

**Alternativas descartadas (não refazer a análise):**
- *Trocar de arquitetura de rede* — já foram testadas 15 variantes; com ~78 amostras de treino
  todas convergem para o mesmo resultado nulo.
- *Optuna com mais trials* — com validação de 19 dias, o Optuna otimiza o ruído do conjunto de
  validação. Mais trials pioram o problema (overfitting na seleção de hiperparâmetros).
- *LSTM/TCN sobre features agregadas* — só ganham com muitas sequências longas. Com ~1.000
  amostras diárias, gradient boosting sobre features agregadas é a escolha certa e poupa semanas.
- *Manter os datasets achatados* (429–851 colunas) — foi a origem do problema; substituídos pela
  abordagem enxuta da célula 113.

**Pontos que estão corretos e devem ser preservados:**
- Divisão sempre cronológica (`shuffle=False`, `TimeSeriesSplit`) — o erro nº 1 em projetos de
  trading foi evitado.
- Features de contexto diário **sem vazamento**: `atr_dia_anterior`, `mm20_diaria`,
  `range_dia_anterior`, `posicao_range_d1` usam `.shift(1)` corretamente, e o `gap` compara a
  abertura do dia (09:00) com o fechamento anterior — ambos conhecidos antes das 11:00. Verificado
  linha a linha.
- A documentação markdown sobre features vs. labels, data leakage e drawdown/run-up está correta.
- Escolha do ticker `WIN$N` (rolagem por liquidez, sem ajuste) está bem fundamentada.

**Suposições assumidas:**
- Os outputs guardados nas células 56, 60 e 73 são de uma versão anterior dos ficheiros de dados
  (ver bug de `NUM_TIMESTEPS` na secção 7); não são reproduzíveis com os CSV atuais.
- O utilizador tem o MetaTrader 5 instalado e com histórico disponível para além de jun/2025 —
  **isto precisa de ser confirmado** antes de investir no passo 1.

## 6. Estado do projeto / ambiente

**Raiz do projeto:** `C:\Users\marcos\Documents\GitHub\IA\predição do mini índice\classificação`
**Raiz do repositório git:** `C:\Users\marcos\Documents\GitHub\IA` (repo multi-projeto; esta pasta
é um de vários projetos independentes dentro dele).

### Ficheiros-chave

| Ficheiro | Papel |
|---|---|
| [Predição do preço_M1.ipynb](Predição do preço_M1.ipynb) | Notebook principal, 115 células, ~40 MB (outputs embutidos) |
| [Predição do preço_M1.BACKUP.ipynb](Predição do preço_M1.BACKUP.ipynb) | Cópia de segurança anterior |
| [WIN$N_M1_BRUTO.csv](WIN$N_M1_BRUTO.csv) | WIN M1 limpo — **2025-06-06 a 2026-02-20, 100.000 linhas (truncado)** |
| [WDO$N_M1_BRUTO.csv](WDO$N_M1_BRUTO.csv) | WDO M1 limpo, mesma origem |
| [WIN_9h-12h.csv](WIN_9h-12h.csv) | WIN filtrado 09:00–11:55 |
| [dataset_final.csv](dataset_final.csv) | Dataset achatado só-WIN: 103 dias × 851 features + 3 labels |
| [dataset_hibrido_win_wdo.csv](dataset_hibrido_win_wdo.csv) | Dataset achatado WIN+WDO: 98 dias × 429 features + 3 labels (gerado com `quantidade_velas = 60`) |
| [labels.csv](labels.csv) | Labels por regra (retorno/pavio) |
| [dados_com_3_labels_3_features.csv](dados_com_3_labels_3_features.csv) | Labels por K-Means (abordagem abandonada) |
| [dados_com_labels_DBSCAN.csv](dados_com_labels_DBSCAN.csv) | Labels por DBSCAN (abordagem abandonada) |
| [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md) | Análise técnica detalhada desta sessão |

Existem também os brutos em H1, M5 e M30 para WIN e WDO na mesma pasta.

### Mapa das células do notebook

- **6–13** — coleta e limpeza de dados (célula 6 usa a API MT5; 9/11/13 leem CSV exportado)
- **15, 17** — geração de labels por clusterização (K-Means / DBSCAN) — **abandonado**
- **19** — geração de labels por regra (retorno > 50 pts, pavio < 250) — **em uso**
- **21, 23** — construção dos datasets achatados (`dataset_final` e `dataset_hibrido`)
- **26–37** — profiling (ydata-profiling), matrizes de correlação, reorganização do treino
- **45–51** — MLP (fixo e com Optuna)
- **54–62** — CNN 1D multi-entrada e TCN
- **66–88** — LSTM e variantes
- **92–96** — XGBoost e SVM
- **98–102** — DBSCAN/PCA exploratório
- **105–108** — teste ADF de estacionariedade e SARIMA
- **113–114** — **abordagem enxuta**: ~15 features agregadas + walk-forward com baseline
  (nunca executadas, sem outputs — é o caminho a seguir)

### Ambiente

- Windows 10, PowerShell. **`git` não está no PATH desta shell** — usar o caminho completo do
  executável ou o cliente gráfico para operações de versionamento.
- Branch git: `main`. Antes desta sessão o estado era limpo (`clean`); os três ficheiros criados
  agora (`CONTEXTO.md`, `CLAUDE.md`, `DOCS/analise-notebook-M1.md`) estão **por commitar**.
- Dependências usadas no notebook: `MetaTrader5`, `pandas`, `numpy`, `scikit-learn`, `tensorflow`,
  `xgboost`, `optuna`, `plotly`, `seaborn`, `matplotlib`, `ydata-profiling`, `statsmodels`,
  `pmdarima`, `keras-tcn`. `TA-Lib` é opcional (o código tem fallback próprio para RSI/MACD).
- Sem variáveis de ambiente ou segredos envolvidos.

## 7. Bloqueios e pendências ⚠️

**RESOLVIDO na sessão 2 — o bloqueio de dados deixou de existir.** A pergunta em aberto era se o
MetaTrader tinha histórico anterior a jun/2025. **Tem:** os ficheiros M30 e H1 já no disco cobrem
desde fevereiro de 2021 (1.248 pregões). O M5, também já no disco, tem 891 amostras úteis — 7,4×
mais do que o M1 usado no notebook. Não é preciso descarregar nada para começar; a exportação dos
2 blocos de M5 apenas leva de 891 para ~1.242.

**Decisão a aguardar do utilizador:** confirmar a reformulação da contribuição do paper
(benchmark metodológico vs. resultado negativo — ver [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md), §2)
e autorizar o início da migração notebook → scripts.

**CONFIRMADO na sessão 3: o resultado é negativo.** O +3,2 p.p. da sessão 2 (amostra truncada,
n = 895) **não sobreviveu** ao alargamento para n = 1.240: caiu para +0,7 p.p., com p = 0,333 no
teste de permutação. Era ruído. A conclusão do paper será, muito provavelmente, um **resultado
negativo** — que é publicável, mas o utilizador deve decidir com esta informação à frente.

**Risco metodológico principal agora:** procurar variantes até uma "funcionar" é garimpagem de
dados e invalida o paper. A defesa está na lista fechada de 5 variantes em
[DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) §7, que deve ser **commitada antes** de ser
corrida, e cujos resultados devem ser **todos** reportados com correção de Holm-Bonferroni.

**Bugs por corrigir** (nenhum foi corrigido nesta sessão):

| Célula | Problema |
|---|---|
| 37 | `try:` com `df = pd.read_csv(...)` indentado a 8 espaços e o resto do bloco a 4 ⇒ **IndentationError**. A célula não corre. |
| 56, 60, 73 | `NUM_TIMESTEPS = 240`, mas o `dataset_hibrido_win_wdo.csv` no disco foi gerado com `quantidade_velas = 60` (429 features = 9 contexto + 7×60) ⇒ **KeyError** em `_c61`. Os outputs guardados são de outra versão do ficheiro. |
| 94 | `evals=[(dtest_final, 'test')]` + `early_stopping_rounds=50` ⇒ o nº de árvores é escolhido a olhar para o conjunto de teste. **Vazamento direto.** |
| 51, 56, 60, 62, 71 | Treino final com `EarlyStopping(monitor='loss', restore_best_weights=True)` — monitorizar a perda de *treino* garante restaurar o ponto de máximo overfitting. |
| 108 | `s#erie_precos = df['vol'].dropna()` ⇒ avalia `s`, **NameError**. |
| 62 | `TimeSeriesSplit(n_splits=20)` sobre ~78 amostras ⇒ folds de validação com ~3 dias. F1-macro sobre 3 amostras não é métrica. |
| 21, 23 | `df_completo.replace(0, 1e-9, inplace=True)` aplica a *todas* as colunas; um `Volume_Tick` zerado faz explodir `tamanho_medio_ordem = Volume_Real / Volume_Tick`. |
| 37, 45, 49 | Balanceamento por `iloc[:min_samples]` descarta os dias **mais recentes** das classes maioritárias e intercala C/V/L de épocas diferentes. Substituir por `class_weight` no `fit()`. |
| geral | `exit()` dentro de notebook mata o kernel. Usar `raise` ou um bloco `if`. |

**Inconsistência de escopo:** o título do notebook e a célula 21 tratam do intervalo 11:00–11:59,
mas a célula 23 (que gera o `dataset_hibrido`, o mais usado) usa features 09:00–09:59 e label
10:00–10:59. Confirmar com o utilizador qual é o alvo oficial.

**Dívida técnica:** o notebook tem 40 MB por causa dos outputs embutidos — inviável em git.

## 8. Comandos úteis

```powershell
# Contar células do notebook
(Get-Content "Predição do preço_M1.ipynb" -Raw -Encoding UTF8 | ConvertFrom-Json).cells.Count

# Extrair só o código (sem outputs) para um ficheiro de trabalho
$nb = Get-Content "Predição do preço_M1.ipynb" -Raw -Encoding UTF8 | ConvertFrom-Json
$sb = New-Object System.Text.StringBuilder; $i = 0
foreach ($c in $nb.cells) {
    if ($c.cell_type -eq 'code') {
        [void]$sb.AppendLine("`n===== CODE CELL $i =====")
        [void]$sb.AppendLine(($c.source -join ""))
    }
    $i++
}
$sb.ToString() | Out-File "code_only.txt" -Encoding utf8

# Verificar cobertura temporal de um CSV bruto
Get-Content "WIN`$N_M1_BRUTO.csv" -TotalCount 2
Get-Content "WIN`$N_M1_BRUTO.csv" -Tail 1

# Abrir o notebook
jupyter notebook "Predição do preço_M1.ipynb"

# Dependências
pip install MetaTrader5 pandas numpy scikit-learn tensorflow xgboost optuna plotly seaborn matplotlib ydata-profiling statsmodels pmdarima keras-tcn
```

Nota: `git` não está no PATH desta shell PowerShell.

## 9. Como retomar

Leia este ficheiro por completo e depois, conforme o que for fazer:
[DOCS/analise-timeframes.md](DOCS/analise-timeframes.md) (dados),
[DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md) (bugs e diagnóstico),
[DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) (requisitos de publicação).

O trabalho parou na **secção 3**. Continue a partir do **passo 1 da secção 4** (exportar o M5
completo em 2 blocos) — não há decisão pendente que o impeça.

**Três avisos para o próximo chat:**

1. **Use o M5, não o M1.** O notebook inteiro está construído sobre o M1, que é o pior ficheiro
   disponível (121 amostras úteis contra 891 do M5). Trocar a fonte é a primeira alteração a fazer
   no código.
2. **Não teste novas arquiteturas antes de o pipeline estar sobre M5 e sem os bugs da secção 7.**
   Já foram testadas 15 variantes sobre dados insuficientes e o resultado foi ruído.
3. **Nenhum resultado de execução única serve para o paper.** Mínimo de 10 sementes com média ±
   desvio-padrão.
