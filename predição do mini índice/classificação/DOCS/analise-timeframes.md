# Análise dos ficheiros brutos do WIN por timeframe

> **ATUALIZAÇÃO 2026-08-12 (sessão 3): o problema de dados foi RESOLVIDO.**
> O utilizador reexportou o M5 do MetaTrader após ajustar "Máx. barras no gráfico".
> Novo ficheiro: `WIN$N_M5_202108120900_202608121830.csv` → convertido para
> [WIN$N_M5_BRUTO_COMPLETO.csv](../WIN$N_M5_BRUTO_COMPLETO.csv).
> **1.240 amostras úteis** (2021-08-12 a 2026-08-12), contra 895 do ficheiro truncado e 121 do M1.
> Validação de fuso: 100,00% de coincidência de preços em 99.999 timestamps comuns com o ficheiro
> antigo — sem desvio horário. Ver §7.
> As seções 1–4 abaixo descrevem o primeiro diagnóstico e ficam como registro histórico.

- **Data:** 2026-08-12
- **Ficheiros:** [WIN$N_M1_BRUTO.csv](../WIN$N_M1_BRUTO.csv), [WIN$N_M5_BRUTO.csv](../WIN$N_M5_BRUTO.csv), [WIN$N_M30_BRUTO.csv](../WIN$N_M30_BRUTO.csv), [WIN$N_H1_BRUTO.csv](../WIN$N_H1_BRUTO.csv)
- **Método:** medição direta com pandas (não estimativa). Script em scratchpad, resultados abaixo.

---

## 1. Cobertura e amostras utilizáveis

"Amostra útil" = dia de pregão com a janela de features (09:00–10:59) **e** a janela de label
(11:00–11:59) completas.

| TF | Linhas | Início | Fim | Anos | Dias distintos | **Amostras úteis** | Preço mín | Preço máx |
|---|---|---|---|---|---|---|---|---|
| M1 | 100.000 | 2025-06-06 | 2026-02-20 | 0,71 | 178 | **121** | 132.335 | 194.400 |
| M5 | 100.000 | 2022-06-10 | 2026-01-19 | 3,61 | 901 | **891** | 96.155 | 168.025 |
| M30 | 23.182 | 2021-02-22 | 2026-02-20 | 4,99 | 1.248 | **1.242** | 96.155 | 194.400 |
| H1 | 11.970 | 2021-03-01 | 2026-02-27 | 4,99 | 1.248 | **1.243** | 96.155 | 197.760 |

**O M1 — o ficheiro em que todo o notebook se baseia — é o pior dos quatro.** Tem 121 amostras
úteis contra 891 do M5 e 1.242 do M30/H1. É 7,4× menos dados que o M5, que já está no disco.

### Porquê

M1 e M5 têm **exatamente 100.000 linhas** — bateram no teto de exportação da interface do
MetaTrader. M30 e H1 não bateram porque 5 anos cabem em menos de 100.000 barras. Não é limitação
de histórico da corretora: **M30 e H1 mostram que existe histórico desde fevereiro de 2021.**

---

## 2. Qualidade dos dados

| TF | Pregão coberto | Barras/dia (mediana) | Maior lacuna entre pregões | Volume_Real = 0 |
|---|---|---|---|---|
| M1 | 09:00 → 18:52 | 565 | 5 dias | 0,00% |
| M5 | 09:00 → 18:50 | 113 | 5 dias | 0,00% |
| M30 | 09:00 → 18:50 | 19 | 5 dias | 0,00% |
| H1 | 09:00 → 18:00 | 10 | 5 dias | 0,00% |

Os quatro ficheiros estão **limpos**: sem volume zerado, sem lacunas anómalas (o máximo de 5 dias
corresponde a fins-de-semana prolongados / feriados). A qualidade não é o problema — a quantidade é.

**Não-estacionariedade forte:** o índice foi de 96.155 a 197.760 em 5 anos (+106%). Confirma que
limiares de label em pontos absolutos (50/250) não são comparáveis ao longo da amostra.

---

## 3. Histórico recuperável

A corretora tem desde ~2021-02, ou seja **~1.248 dias de pregão** é o **teto absoluto** desta série.

| TF | Barras para 5 anos | Blocos de exportação necessários |
|---|---|---|
| M1 | ~673.920 | **7** |
| M5 | ~134.784 | **2** |
| M30 | ~22.464 | 1 (já completo) |
| H1 | ~11.232 | 1 (já completo) |

**Recuperar o M5 completo custa 2 exportações** e leva de 891 → ~1.242 amostras. É o melhor
retorno por esforço de todo o projeto.

---

## 4. Adequação de cada timeframe à tarefa

A janela de features é 09:00–10:59 (2 horas). Quantos candles cada TF gera nessa janela:

| TF | Candles na janela | Amostras | Razão amostra/feature (achatado, 7 feats/candle) | Razão (enxuta, 16 feats) |
|---|---|---|---|---|
| M1 | 120 | 121 | 849 features → **0,14 : 1** | 7,6 : 1 |
| M5 | 24 | 891 | 177 features → **5,0 : 1** | 55,7 : 1 |
| M30 | 4 | 1.242 | 37 features → 33,6 : 1 | 77,6 : 1 |
| H1 | 2 | 1.243 | 23 features → 54,0 : 1 | 77,7 : 1 |

**Conclusão: o M5 é o timeframe correto.**

- **M1** — inviável. Mais resolução, mas 121 amostras contra 849 features.
- **M5** — o compromisso certo. 24 candles dão uma sequência com estrutura suficiente para
  CNN/LSTM/TCN, e 891 (→1.242) amostras tornam o treino possível. Tensor `(1242, 24, 7)`.
- **M30 / H1** — mais amostras, mas 4 e 2 candles na janela. **Não servem para arquiteturas
  sequenciais** (não há sequência). Servem bem para (a) features de contexto diário e (b) estender
  o histórico de labels até 2021.

---

## 5. Teste de sinal preliminar (M5, 891 dias)

Aplicada a lógica das células 113/114 (15 features agregadas + walk-forward de 5 folds,
GradientBoosting vs. `DummyClassifier`), com dois esquemas de rotulagem.

### Label atual (limiares fixos: retorno > 50 pts, pavio < 250 pts)

Distribuição: Venda 318 | Compra 297 | Lateral 280 — **naturalmente equilibrada com 5 anos de
dados** (com 98 dias não estava).

| Fold | Teste | Baseline acc | GB acc | Δ | GB F1-macro |
|---|---|---|---|---|---|
| 1 | 149 | 0,309 | 0,403 | **+0,094** | 0,399 |
| 2 | 149 | 0,423 | 0,356 | −0,067 | 0,311 |
| 3 | 149 | 0,342 | 0,423 | **+0,081** | 0,406 |
| 4 | 149 | 0,342 | 0,383 | +0,040 | 0,378 |
| 5 | 149 | 0,295 | 0,309 | +0,013 | 0,308 |
| **Média** | | **0,342** | **0,374** | **+0,032** | **0,360** |

**Interpretação honesta:** o ganho médio (+3,2 p.p.) passa por pouco o limiar de 0,03 da célula
114, mas **não é conclusivo**:

- Um dos cinco folds é **negativo** (−0,067) e outro é praticamente nulo (+0,013).
- Com 745 previsões de teste no total, o erro padrão da acurácia é ~1,8 p.p. ⇒ +3,2 p.p. é
  **~1,8 sigma**. Fica abaixo do limiar convencional de significância.
- As classes estão quase equilibradas, então o acaso puro dá ~33,3%. O GB deu 37,4% — 4 p.p.
  acima do acaso.

**Veredicto: há um indício fraco e inconsistente de sinal, não uma descoberta.** É exatamente o
tipo de resultado que precisa de mais dados (os 1.242 dias) e de teste estatístico formal antes de
qualquer afirmação.

### Label normalizado por ATR (tentativa: retorno > 0,30 × ATR, pavio < 1,5 × ATR)

Distribuição: Lateral 733 | Venda 91 | Compra 71 — **desequilibrada, parametrização má.**

Ganho médio: **−0,044** (sem sinal). O baseline sobe para 82% simplesmente por prever sempre
"Lateral".

**Diagnóstico do erro:** usei o ATR **diário** para normalizar um movimento de **1 hora**. O ATR
diário é grande demais, então quase tudo cai em "Lateral". A correção é usar a volatilidade da
própria janela 11:00–11:59 (ex.: desvio-padrão histórico do retorno dessa hora), não o ATR do dia
inteiro. **Isto invalida o teste do label-ATR, não a ideia de normalizar** — a ideia continua
correta e precisa de ser re-parametrizada.

---

## 6. Recomendações desta análise

1. **Trocar o M1 pelo M5** como fonte primária. Ganho imediato: 121 → 891 amostras, sem
   descarregar nada.
2. **Recuperar o M5 completo** (2 blocos de exportação, ou `mt5.copy_rates_range` em 2 chamadas)
   ⇒ ~1.242 amostras. Meta realista e é o teto da série.
3. **Usar M30/H1 para features de contexto diário**, aproveitando que já cobrem 2021–2026.
4. **Re-parametrizar o label normalizado** com volatilidade da janela-alvo, não ATR diário.
5. **Não usar o pipeline achatado com M5** (5:1 ainda é fraco). Para arquiteturas sequenciais usar
   o tensor 3D `(n, 24, 7)`; para modelos tabulares usar a abordagem enxuta da célula 113.

---

## 7. Recuperação do histórico M5 — concluída (2026-08-12)

O utilizador aplicou o **Método A** (MetaTrader → Ferramentas → Opções → Gráficos →
"Máx. barras no gráfico" = Ilimitado → reiniciar → `Home` no gráfico → exportar) e obteve
`WIN$N_M5_202108120900_202608121830.csv` (8,4 MB, separado por tabulação, UTF-8, colunas
`<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`).

Convertido para o formato do projeto em
[WIN$N_M5_BRUTO_COMPLETO.csv](../WIN$N_M5_BRUTO_COMPLETO.csv)
(`datetime;Abertura;Fechamento;Maxima;Minima;Volume_Real;Volume_Tick`).

### Comparação

| | M1 (notebook) | M5 truncado | **M5 completo** |
|---|---|---|---|
| Período | 2025-06 → 2026-02 | 2022-06 → 2026-01 | **2021-08 → 2026-08** |
| Barras | 100.000 | 100.000 | **138.400** |
| Dias distintos | 178 | 901 | **1.248** |
| **Amostras úteis** | **121** | **895** | **1.240** |

Ganho face ao M5 truncado: **1,39×**. Face ao M1 usado no notebook: **10,2×**.

### Validação de integridade

- **Fuso horário:** 99.999 timestamps em comum com o ficheiro antigo, **100,00%** de coincidência
  nos preços de abertura. Confirmado que a exportação nova está no mesmo fuso — os dois ficheiros
  são intercambiáveis.
- **Pregão coberto:** 09:00 → 18:50, idêntico ao anterior.
- **Confirmação do teto da série:** 1.248 dias distintos, exatamente o mesmo número medido no
  M30 e no H1. **Confirma que ~1.248 pregões é o limite real do histórico desta corretora** — não
  há mais dados a recuperar. Este é o n máximo do estudo e deve ser declarado como limitação da
  amostra no paper.

### Razão amostra/feature com o M5 completo

| Abordagem | Features | Razão |
|---|---|---|
| Enxuta (célula 113) | 15 | **83 : 1** — saudável |
| Sequencial 3D `(1240, 24, 7)` | 168 valores/amostra | 7,4 : 1 — viável para CNN/LSTM pequenas |
| Achatada (pipeline antigo) | 177 | 7,0 : 1 — marginal, não recomendada |

---

## 8. Teste de sinal definitivo com n = 1.240 — RESULTADO NULO

Walk-forward de 5 folds (206 dias de teste cada), GradientBoosting vs. `DummyClassifier`,
15 features agregadas, razão 83:1. Dois esquemas de rotulagem.

### A — Label fixo da ideia primária (retorno > 50 pts, pavio < 250 pts)

Distribuição: Lateral 428 | Venda 424 | Compra 388

| Fold | Baseline | GB | Δ |
|---|---|---|---|
| 1 | 0,359 | 0,340 | −0,019 |
| 2 | 0,374 | 0,393 | +0,019 |
| 3 | 0,383 | 0,345 | −0,039 |
| 4 | 0,330 | 0,296 | −0,034 |
| 5 | 0,238 | 0,345 | +0,107 |
| **Média** | **0,337** | **0,344** | **+0,007** |

**3 dos 5 folds são negativos.**

### B — Label normalizado pela volatilidade da própria janela

Escala = mediana móvel de 60 dias de `|retorno 11h|`, com `shift(1)`. Calibrada para equivaler a
~50 pts em média (`k_ret = 0,16`, `k_pav = 0,81`).

Distribuição: Venda 427 | Lateral 412 | Compra 401 — **bem equilibrada** (ao contrário da tentativa
falhada com ATR diário, §5).

| Fold | Baseline | GB | Δ |
|---|---|---|---|
| 1 | 0,306 | 0,311 | +0,005 |
| 2 | 0,350 | 0,296 | −0,053 |
| 3 | 0,335 | 0,345 | +0,010 |
| 4 | 0,335 | 0,316 | −0,019 |
| 5 | 0,306 | 0,330 | +0,024 |
| **Média** | **0,326** | **0,319** | **−0,007** |

### Teste de permutação (label A, 200 permutações)

| | Valor |
|---|---|
| Ganho observado | **+0,0068** |
| Ganho sob hipótese nula (média) | −0,0003 |
| Desvio-padrão do nulo | 0,0160 |
| Percentil 95 do nulo | +0,0263 |
| **z** | **0,44** |
| **p-valor** | **0,333** |

**Não significativo a 5%.** O ganho observado está a menos de meio desvio-padrão do acaso.

### Conclusão

**Não há sinal detetável.** Com a amostra completa, o ganho sobre o baseline é +0,7 p.p. — dentro
do ruído. O ganho de +3,2 p.p. medido na amostra truncada (895 dias, §5) **não sobreviveu** ao
alargamento da amostra: era ruído, como o próprio teste de permutação agora confirma.

Dois pontos importantes:

1. **A redefinição do label não é a solução.** A hipótese anterior era que o label fixo, com
   limiares absolutos, estaria a destruir o sinal. O label B corrige isso — limiares adaptativos,
   classes equilibradas — e o resultado é igualmente nulo (−0,007). Isto **enfraquece a hipótese
   do label** como causa principal.
2. **Escala relevante:** a mediana de `|retorno|` na janela 11:00–11:59 é de **310 pontos**. O
   limiar de 50 pontos do label A é 0,16× isso, ou seja, quase todos os dias o ultrapassam. Na
   prática, **o label A é determinado quase só pelo filtro `pavio < 250`**, não pela direção —
   "Lateral" significa efetivamente "dia de muito ruído intradiário", não "dia parado".

---

## 9. Escolha da janela horária — exploração e recomendação

**Protocolo:** o *sweep* correu apenas nos **primeiros 80% dos dias** (998 dias, até 2025-08-12).
Os últimos 250 dias ficaram intocados como holdout. Label por **tercis** (quantis móveis de 250
dias com `shift(1)`), o que equilibra as classes por construção e torna a acurácia comparável
entre configurações — o baseline fica sempre próximo de 1/3.

| Entrada | Alvo | n | Baseline | GB | **Ganho** | dp entre folds | Nota |
|---|---|---|---|---|---|---|---|
| 09:00–10:59 | 11:00–11:59 | 893 | 0,322 | 0,361 | **+0,039** | 0,039 | atual do notebook |
| 09:00–11:29 | 11:30–12:29 | 893 | 0,349 | 0,355 | +0,007 | 0,060 | abertura EUA (V3) |
| 10:00–10:59 | 11:00–11:59 | 877 | 0,322 | 0,340 | +0,018 | 0,064 | entrada curta 1h |
| 09:00–09:59 | 10:00–10:59 | 870 | 0,337 | 0,310 | −0,026 | 0,044 | 1ª hora → 2ª hora |
| 09:00–10:59 | 11:00–12:59 | 893 | 0,350 | 0,353 | +0,003 | 0,048 | alvo longo 2h |
| 09:00–11:59 | 12:00–12:59 | 893 | 0,292 | 0,343 | **+0,051** | 0,085 | entrada 3h |
| 09:00–10:59 | 15:00–15:59 | 893 | 0,361 | 0,311 | −0,050 | 0,021 | **controlo** |

- Média dos ganhos: **+0,006**
- Desvio-padrão entre configurações: **0,033**

### Interpretação

**Nenhuma janela é demonstravelmente melhor.** O argumento decisivo é aritmético: se os 7 ganhos
forem ruído centrado em zero com dp = 0,033, o **máximo esperado de 7 extrações** é
≈ 0,033 × 1,35 ≈ **+0,045**. O máximo observado foi **+0,051**. Está exatamente onde o acaso o
colocaria.

Além disso, em **todas** as configurações o desvio-padrão do ganho **entre folds** é da mesma
ordem ou maior do que o próprio ganho — nenhuma é estável.

O **controlo funcionou**: prever a janela 15:00–15:59 a partir da manhã dá −0,050. O arnês deteta
corretamente a ausência de sinal, o que dá confiança de que os números acima não são artefacto.

### Recomendação

**Manter `09:00–10:59 → 11:00–11:59`** como configuração primária, por quatro razões:

1. **É a configuração da ideia primária.** Trocá-la com base neste sweep seria escolher o máximo de 7
   testes — precisamente a garimpagem de dados que invalida o paper.
2. **É a mais estável entre as positivas** (dp entre folds 0,039 contra 0,085 da "entrada 3h").
3. **Preserva a tese económica**: entrar às 11:00 e sair às 12:00.
4. **Maximiza n** (893 no sweep, 1.220 na amostra completa).

**Única alternativa que merece ser testada no holdout:** `09:00–11:59 → 12:00–12:59`. Não por ter
tido o maior ganho, mas porque tem **justificação causal independente** — a janela de entrada
inclui as 11:30, a abertura americana, que é o mecanismo apontado no próprio notebook M1 como
driver do movimento. Um resultado positivo aí seria interpretável; nas outras, seria só sorte.

**Regra:** o holdout aceita **uma** configuração. Escolha antes de o abrir.

---

## 10. Erro de implementação descoberto no sweep

Duas configurações falharam inicialmente com "0 amostras". A causa: **`RSI(14)` numa janela de
1 hora em M5 (12 candles) é inteiramente `NaN`**, e o `dropna()` seguinte eliminava todas as
linhas. Silencioso e fatal.

Corrigido com período adaptativo — `n = max(2, min(14, len(c) // 2))` — já incorporado no notebook
M5. O `MACD` não sofre do mesmo problema porque `ewm` não gera `NaN`.

Fica o aviso: **qualquer janela mais curta que ~28 candles exige rever os períodos dos
indicadores.**
   limiar de 50 pontos do label A é 0,16× isso, ou seja, quase todos os dias o ultrapassam. Na
   prática, **o label A é determinado quase só pelo filtro `pavio < 250`**, não pela direção —
   "Lateral" significa efetivamente "dia de muito ruído intradiário", não "dia parado". Isto é um
   achado sobre o desenho do label que vale reportar.
