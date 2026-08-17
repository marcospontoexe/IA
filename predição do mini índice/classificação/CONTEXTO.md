# CONTEXTO DA SESSÃO

- **Última atualização:** 2026-08-16 19:05
- **Sessão nº:** 4
- **Status geral:** em andamento — notebook M5 completo e validado (42 células, 15 seções);
  **resultado direcional é NULO e está bem sustentado**; falta rodar a bateria de variantes

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

## 3. Em andamento 🔧

Nada em execução.

**Ponto exato de parada:** terminei de revisar as arquiteturas do notebook antigo e registrar o
que valia a pena. Nesta sessão entraram, em ordem:

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

**O resultado continua nulo**, agora com muito mais sustentação do que na sessão 3.

**Estado do holdout:** os 244 dias finais (2025-08-18 a 2026-08-12) **nunca foram tocados**.
`EXECUTAR_HOLDOUT = False` de propósito. Ver o aviso na seção 4 abaixo.

## 4. Próximos passos (planejado) 📋

1. **Rodar o notebook do começo com as configurações finais.** As features e as redes mudaram
   várias vezes; os números que estão no [README.md](README.md) são de uma rodada antiga
   (15 features, entrada única, custo 3,0) e **estão desatualizados**. Subir `SEMENTES` para 10 e
   `N_SEMENTES_DL` para 10.
2. **Atualizar o README** com os números da rodada nova.
3. **Rodar a bateria de 5 variantes** ([DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) §7).
   Começar pela **V4 (prever volatilidade em vez de direção)** — é a de maior probabilidade a
   priori, e três achados independentes já apontam para ela:
   - as únicas features que separam as classes são de volume/atividade (seção 7)
   - o contexto diário (medidas de volatilidade recente) bate as features da janela (seção 15.1)
   - a autocorrelação da volatilidade realizada é positiva; a da direção não é
4. **Testar ROCKET / MiniROCKET** (seção 9). É o método com melhor perfil para 969 amostras:
   kernels aleatórios não treinados + classificador linear.
5. **Só então abrir o holdout**, com **uma única** configuração: `09:00–11:59 → 12:00–12:59`,
   escolhida por justificativa causal (inclui a abertura americana das 11:30), não por ter dado
   o maior ganho no sweep.
6. **Migrar para scripts** com semente parametrizável, para o paper.
7. **Análise estatística formal** com correção de Holm-Bonferroni sobre todas as variantes.

> **Aviso sobre o holdout.** Com 244 dias, o erro padrão da acurácia é de 3,0 p.p. — ele só
> detecta ganho **acima de ~8 p.p.**. O ganho observado é de ~0,6 p.p. Ele serve para confirmar o
> resultado nulo ou detectar um sinal grande; **não tem poder para validar um sinal marginal**.
> Isso está na seção 12 e precisa constar no paper como limitação.

## 5. Decisões e raciocínio 🧠

### O resultado, e o que o sustenta

| Evidência | Resultado |
|---|---|
| Melhor modelo vs baseline (walk-forward, 10 sementes) | ~+0,6 p.p. |
| Teste de permutação (200 embaralhamentos) | **p = 0,33** |
| Friedman entre todos os modelos | p = 0,079 — sem diferença detectável |
| Backtest out-of-fold com custo de 6 pts | **negativo** |
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
| [classificação Compra-Lateral-Venda do WIN M5.ipynb](classificação%20Compra-Lateral-Venda%20do%20WIN%20M5.ipynb) | **Notebook ativo.** 42 células, 20 de código, 15 seções |
| [WIN$N_M5_BRUTO_COMPLETO.csv](WIN$N_M5_BRUTO_COMPLETO.csv) | **Base principal.** 138.400 barras, 1.240 amostras úteis |
| `WIN$N_{M1,M5,M30,H1}_2021...csv` | exportações brutas do MetaTrader, todas 2021-08 a 2026-08 |
| [coletar_m5.py](coletar_m5.py) | coleta via API do MT5 (alternativa à exportação manual) |
| [README.md](README.md) | visão geral e roteiro — **números desatualizados** |
| [Predição do preço_M1.ipynb](Predição do preço_M1.ipynb) | notebook antigo, **a apagar** |

### Estrutura do notebook ativo

```
 1. Configuração              9. Modelos sequenciais (deep learning)
 2. Carregamento e validação 10. Configuração das redes (11 subseções)
 3. Engenharia de features   11. Comparação estatística
 4. Rotulagem                12. Avaliação econômica
 5. Protocolo de avaliação   13. Holdout final
 6. Tratamento dos dados     14. Registro da rodada
 7. Análise exploratória     15. Conclusão
 8. Modelos tabulares        + O que vem depois  + Observações
```

**Configuração atual:** janela `09:00–10:59 → 11:00–11:59`, rótulo `vol`, 21 features
(16 janela + 5 contexto), 976 dias de trabalho + 244 de holdout, custo 6,0 pts.

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

**Dívida menor:** o [README.md](README.md) tem a tabela de 11 modelos com números de uma rodada
antiga (15 features, entrada única, custo 3,0). Atualizar depois da rodada nova.

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

Leia este arquivo, depois o notebook ativo — em especial a **seção 15 (Conclusão)**, que
consolida tudo com os números.

**Continue pelo passo 1 da seção 4:** rodar o notebook do começo com `SEMENTES = 10` e
`N_SEMENTES_DL = 10`, e atualizar o README com os números novos.

**Cinco avisos:**

1. **Valide com o anaconda, não com o Python do PATH.** O do PATH não tem TensorFlow e pula a
   seção 9 em silêncio.
2. **Não abra o holdout** antes de fechar todas as decisões. Uma configuração, uma vez.
3. **Não teste arquitetura nova** esperando que resolva. Onze modelos, quatro timeframes, sete
   janelas — tudo no ruído. O gargalo não é o modelo.
4. **Não corte features** olhando a tabela de informação mútua. Está medido: é pior que sortear.
5. **Não otimize por macro F1.** Correlação −0,536 com o P&L. Se houver busca de
   hiperparâmetros, o alvo é o resultado econômico out-of-fold.
