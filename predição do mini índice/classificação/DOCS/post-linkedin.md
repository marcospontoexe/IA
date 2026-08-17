# Rascunhos de post para o LinkedIn

Objetivo: atrair atenção de recrutadores e gestores técnicos, usando o resultado nulo como
demonstração de rigor metodológico.

**A tese de posicionamento:** a habilidade escassa em ML aplicado a mercado não é treinar
modelo, é não se enganar. O post inteiro é a demonstração disso.

---

## Rascunho A — "quase me enganei" (recomendado)

> É o de maior chance de engajamento. Tem tensão narrativa, um número que parece bom, e a
> reviravolta. Termina em competência, não em fracasso.

---

Passei cinco meses construindo um modelo para prever o mini-índice.

Ele não funciona. E o trabalho mais difícil foi provar isso.

Testei 11 modelos em 1.240 pregões: regressão logística, SVM, Random Forest, XGBoost, MLP,
CNN 1D, LSTM e TCN. Validação cronológica, 10 sementes, custo de operação medido no book, não
chutado.

Nenhum bateu de forma consistente o modelo trivial de chutar sempre a classe mais comum.

Aí abri o conjunto de teste final. Os 244 dias que eu tinha separado no começo do projeto e
nunca tinha tocado.

Ganho de 5,7 pontos percentuais sobre o baseline. R$ 1.167 de lucro simulado.

Por um minuto, achei que tinha alguma coisa.

Três testes depois, não tinha:

1. Antes de abrir o conjunto, eu tinha calculado o ganho mínimo detectável com 244 dias:
   8,4 p.p. O meu foi 5,7. Abaixo do próprio limiar que eu mesmo fixei.

2. A classe mais frequente mudou entre treino e teste. Isso rebaixou o baseline sozinho.
   36% do meu "ganho" era deriva de distribuição, não habilidade do modelo.

3. Simulei 20 mil estratégias aleatórias com o mesmo número de operações. O meu modelo caiu
   no percentil 78. E no walk-forward, a estratégia aleatória rendeu +44 pontos por operação
   contra +36 do modelo.

O acaso ganhou do meu modelo. E eu só descobri porque fui procurar.

O que me salvou de publicar isso como descoberta não foi esperteza. Foi ter escrito a regra
de decisão ANTES de olhar o número, e ter me obrigado a cumprir.

Prever a direção do índice nessa janela não funciona. Isso é um resultado, e está publicado
inteiro: código, dados, os testes que derrubam meu próprio trabalho, e o que vem depois.

Link nos comentários.

#MachineLearning #DataScience #Quant #Python #CienciaDeDados

---

## Rascunho B — metodologia primeiro

> Mais sóbrio, menos alcance, mais filtro. Bom se o alvo for time de pesquisa ou quant desk,
> onde o público lê o detalhe.

---

Publiquei um resultado negativo. Cinco anos de dados do mini-índice, 11 modelos, nenhum
prevê direção melhor que o acaso.

O que eu acho que vale mais que o resultado é o protocolo que me impediu de concluir o
contrário:

→ Divisão sempre cronológica. Embaralhar série temporal deixa o modelo ver o futuro.

→ Baseline em todo teste. 36% de acurácia em 3 classes parece razoável até você medir que
chutar a classe majoritária dá 35%.

→ Conjunto de teste final aberto uma vez só, com a regra de decisão escrita antes.

→ Teste de permutação com 200 embaralhamentos. É o que separa sinal de sorte: p = 0,144.

→ Friedman + Nemenyi entre os modelos. Nenhum par difere. O baseline terminou em 3º de 7.

→ Correção de Bonferroni nas 21 features. Zero sobrevivem.

→ Backtest com custo medido (7 pontos por operação, spread real do ativo). Negativo.

→ Semente declarada em tudo, inclusive no TensorFlow. Rodei duas vezes do zero e os números
saíram idênticos até a quarta casa.

O teste final mostrou +5,7 p.p. e lucro simulado. Parecia bom. Não era: abaixo do poder
estatístico do conjunto, com 36% do ganho vindo de deriva de classe, e no percentil 78 de
20 mil estratégias aleatórias.

Prefiro publicar isso do que uma acurácia de 70% que ninguém consegue reproduzir.

Repositório completo nos comentários, com o notebook inteiro comentado.

#MachineLearning #DataScience #Quant #Reprodutibilidade #Python

---

## O comentário com o link

Publique o link no PRIMEIRO comentário, logo depois de postar. O LinkedIn reduz o alcance de
post com link externo no corpo.

> Repositório: github.com/<usuário>/<repo>
>
> O notebook tem 17 seções, cada decisão metodológica justificada e medida. As partes que eu
> acho mais úteis para quem trabalha com série temporal são a 5 (protocolo), a 13 (o teste
> final e os três testes que derrubam o resultado) e a 15 (conclusão com os números).

---

## A imagem

Uma imagem só, e a mais forte é o contraste:

**Curva de resultado do holdout subindo até +5.838 pontos, ao lado do histograma das 20 mil
estratégias aleatórias com a linha do modelo caindo no percentil 78.**

Legenda: *"À esquerda, parece que funcionou. À direita, por que não funcionou."*

É autoexplicativa, cabe no feed e conta a história sem o texto.

Alternativa mais simples: a tabela dos 11 modelos com a linha do baseline destacada no meio,
mostrando que 4 modelos reais ficam abaixo dele.

---

## Antes de publicar: arrumar a casa

Quem se interessar vai clicar. O que a pessoa encontra hoje:

| Item | Situação | O que fazer |
|---|---|---|
| Caminho do projeto | `IA/predição do mini índice/classificação` | link direto para a pasta, ou repositório próprio |
| `CONTEXTO.md` | handoff interno de sessão, exposto na raiz | mover para `DOCS/` ou explicar no README que é diário de bordo |
| `CLAUDE.md` | instruções de ferramenta | mover para `.github/` ou `DOCS/` |
| `Predição do preço_M1.ipynb` | notebook antigo, marcado para apagar | apagar antes de divulgar |
| `rascunho.md` | anotações soltas | apagar ou mover |
| Idioma | tudo em pt-BR | ok para o mercado brasileiro; para vagas internacionais, um README curto em inglês ajuda |

O README já está bom e é a porta de entrada certa. Ele abre com o resultado, mostra a tabela
e explica o método, que é a ordem certa para quem tem 40 segundos.

---

## Mecânica do post

- **As duas primeiras linhas são tudo.** No celular o resto fica atrás do "ver mais". As duas
  primeiras linhas do rascunho A já entregam a tensão.
- **Parágrafos de 1 a 2 linhas**, com linha em branco entre eles. Bloco denso ninguém lê.
- **Sem link no corpo.** Primeiro comentário.
- **3 a 5 hashtags**, no fim.
- **Terça a quinta, de manhã.** Segunda e sexta rendem menos.
- **Responda todo comentário nas primeiras duas horas.** É o que sustenta o alcance.
- **Não escreva "estou em busca de oportunidades" no post.** Deixe o trabalho falar e mantenha
  o perfil com o "Open to work" ligado. Pedido explícito no meio de um post técnico corta a
  credibilidade que o próprio post construiu.

## Prepare-se para dois tipos de comentário

**"Você devia ter tentado X."** Vai aparecer, e é bom para o alcance. A melhor resposta é
mostrar que você já testou e medir: a lista de variantes está pré-registrada no repositório,
e várias sugestões comuns (mais features, outra arquitetura, outro timeframe) já foram
testadas e estão documentadas com o número.

**"Modelo de mercado não funciona mesmo."** Concorde em parte e limite o escopo: o que foi
testado é direção nessa janela específica, com essas features, nesse ativo. Não é uma
afirmação sobre mercados em geral, e o próprio repositório lista o que ainda falta testar.

## O que não fazer

- Não escreva "descobri que o mercado é imprevisível". É grande demais para o que foi medido,
  e um leitor técnico desconta na hora.
- Não omita que o resultado é nulo para gerar clique. A reviravolta é o valor.
- Não use os R$ 1.167 sem a explicação seguinte. Fora de contexto vira promessa de retorno.
