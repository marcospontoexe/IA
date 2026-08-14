# Projeto: Classificação do mini-índice (WIN)

Classificador que prevê o comportamento do mini-índice na janela das 11:00–11:59
(**Compra** / **Venda** / **Lateral**) a partir dos candles das 09:00–10:59.

> **Comece sempre por ler [CONTEXTO.md](CONTEXTO.md)** — é o handoff entre sessões e diz onde o
> trabalho parou.

---

## Índice de documentação

| Tópico | Ficheiro |
|---|---|
| Visão geral do projeto e roteiro de experimentos | [README.md](README.md) |
| **Notebook ativo (M5)** | [classificação Compra-Lateral-Venda do WIN M5.ipynb](classificação%20Compra-Lateral-Venda%20do%20WIN%20M5.ipynb) |
| Estado da sessão / handoff | [CONTEXTO.md](CONTEXTO.md) |
| Análise técnica do notebook: diagnóstico, bugs e plano de correção | [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md) |
| Análise dos ficheiros brutos por timeframe + teste de sinal preliminar | [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md) |
| Protocolo experimental para publicação científica | [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md) |

---

## Regras específicas deste projeto

- **Não testar novas arquiteturas de modelo** antes de resolver o problema de quantidade de dados.
  Já foram testadas ~15 variantes (MLP, CNN, TCN, LSTM, XGBoost, SVM) e todas ficaram no ruído.
  O gargalo é a razão amostra/feature e a definição do label, não o modelo. Detalhe em
  [DOCS/analise-notebook-M1.md](DOCS/analise-notebook-M1.md).
- **Qualquer resultado de modelo deve ser comparado a um baseline** (`DummyClassifier` com
  `strategy='most_frequent'`) em walk-forward cronológico. Acurácia sem baseline não é resultado.
- **Divisão de dados sempre cronológica** (`shuffle=False`, `TimeSeriesSplit`). Nunca embaralhar.
- **Timeframe primário é o M5**, não o M1. O M1 tem 121 amostras úteis; o M5 tem 891. Medição em
  [DOCS/analise-timeframes.md](DOCS/analise-timeframes.md).
- **Qualquer resultado destinado ao paper exige ≥10 sementes** e média ± desvio-padrão. Execução
  única não é resultado. Ver [DOCS/protocolo-paper.md](DOCS/protocolo-paper.md).
- `git` não está no PATH da shell PowerShell deste ambiente.
