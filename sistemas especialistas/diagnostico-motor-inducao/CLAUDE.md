# CLAUDE.md

Este arquivo orienta o Claude Code ao trabalhar nesta pasta.

## O que é

Sistema especialista em Python que diagnostica falhas de um motor de indução trifásico de 30 kW, que aciona uma bomba centrífuga, e recomenda ações de manutenção. É um projeto de portfólio: o [README.md](README.md) é a documentação para quem visita o repositório. Escreva sempre em pt-BR, e mantenha o tom de projeto (sem referências a curso, atividade ou entrega).

## Arquivos

- [base_conhecimento.json](base_conhecimento.json): 12 entradas, 13 sintomas, 7 diagnósticos, 9 ações e 19 regras. A negação nas regras é escrita `"NAO S03"` e só vale para sintomas, para a inferência continuar monotônica.
- [sistema_especialista.py](sistema_especialista.py): encadeamento para frente com lógica de três valores (presente, ausente, desconhecido). Quando faltam dados, lista hipóteses com a entrada que falta. Só usa a biblioteca padrão e roda no Python 3.13 e 3.14 e num kernel IPython, como o do Spyder.
- [testes.py](testes.py): 13 casos, 3 verificações de validação e a conferência de que todas as regras disparam. Termina com código 1 se algo falhar.
- [leitura_exemplo.json](leitura_exemplo.json): leitura simulada do "Motor M-101", que é o caso T11 dos testes e o exemplo do README.

## Comandos

```powershell
python -B testes.py                                  # ou -v para os relatórios completos
python -B sistema_especialista.py leitura_exemplo.json
python sistema_especialista.py                       # modo interativo
```

- Para ver a saída com acentos e Ω, redirecione pelo `cmd`, porque o script grava em UTF-8: `cmd /c "python testes.py > <scratchpad>\saida.txt"`. O redirecionamento do PowerShell estraga os caracteres.
- O `-B` evita deixar `__pycache__` na pasta (o `.gitignore` também o exclui).

## Manter o README sincronizado

O [README.md](README.md) reproduz as tabelas da base (seções "Sintomas" e "Base de conhecimento"), os resultados dos testes (seção "Testes"), a saída de `python sistema_especialista.py leitura_exemplo.json` (seção "Exemplo de uso", que precisa ser idêntica) e o bloco de hipóteses do caso T13. Se a base ou o código mudarem, rode os comandos de novo e atualize o README.

## Origem dos limites

- S01 e S02 vêm da ISO 10816-3, S11 da NEMA MG 1 e S13 da IEEE 43.
- S08 e S09 foram adotados no projeto; o S09 parte dos 155 °C da classe F (IEC 60085).
- S03 e S12 são regras práticas, e S10 é a corrente de placa.
- O equipamento (30 kW, base rígida, classe F, acionando uma bomba) define esses limites: grupo 2 da ISO 10816-3 e classe térmica F.
- Os testes verificam o código contra a base, mas não validam a base contra motores reais. O README diz isso.
