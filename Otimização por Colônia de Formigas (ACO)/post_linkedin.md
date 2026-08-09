# Post do LinkedIn — texto e roteiro de publicação

## Texto do post

> Copie do bloco abaixo. O LinkedIn não aceita markdown: negrito e itálico não funcionam, então o
> texto já está escrito para ser colado como está.

---

Implementei duas meta-heurísticas para otimizar um centro de distribuição. Uma delas foi
praticamente inútil — e descobrir por quê foi a parte mais valiosa do projeto.

O contexto: até 55% da despesa operacional de um armazém vem da separação de pedidos (De Koster et
al., 2007). E a maior fatia disso é o separador simplesmente andando entre os endereços.

O que construí, do zero em Python, sem biblioteca de otimização pronta:

→ modelagem do armazém como grafo (505 nós) + Floyd–Warshall para as distâncias reais
→ Algoritmo Genético para decidir quais pedidos vão no mesmo lote
→ Colônia de Formigas (ACO) para decidir a ordem de visita

Resultado: 14,8% menos distância percorrida, estável em 8 réplicas independentes.

Só que eu não comparei apenas "antes x depois". Rodei um experimento fatorial com as 4 combinações
possíveis, para isolar a contribuição de cada peça. E aí veio a surpresa:

• ACO sozinho: −13,2%
• AG sozinho: −3,6%
• Os dois juntos: −13,7%

O ACO entregava 96% do ganho. O algoritmo genético quase não se pagava.

A causa era uma decisão de projeto minha. Para não rodar o ACO milhares de vezes dentro do laço
evolutivo, usei uma heurística barata como função de aptidão. Só que isso fez o AG otimizar um
critério que o roteirizador final não usava — ele estava mirando o alvo errado.

Aquelas execuções a mais mudaram a recomendação inteira: implantar primeiro o roteirizador, que tem
menor custo de integração e entrega quase todo o retorno.

A lição que levo: comparar antes e depois te dá um número. Isolar as variáveis te dá uma decisão.

O cenário é fictício e os dados são simulados, mas dimensionados a partir de operação real de
varejo. Notebook completo e executável no GitHub — link nos comentários.

#Python #Otimizacao #PesquisaOperacional #Logistica #InteligenciaArtificial

---

## Roteiro de publicação

**1. Antes de postar**

- [ ] Criar o repositório **dedicado** no GitHub (não apontar para a pasta dentro do repo da
      faculdade — um diretório chamado "22-IA na Gestão de Negócios" entrega "trabalho de aula" e
      derruba o clique).
- [ ] Subir: `otimizacao_picking.ipynb`, `README.md`, `resultados.json` e a pasta `figuras/`.
- [ ] Conferir se o notebook renderiza com os gráficos direto no GitHub.

**2. Ao postar**

- Suba o `carrossel.pdf` como **documento** — o LinkedIn distribui melhor esse formato que imagem
  solta. Se preferir imagens, use os `slide1.png` … `slide5.png`.
- Ponha o link do repositório **no primeiro comentário**, não no corpo do post.
- Responda todo comentário nas primeiras horas.

**3. Depois**

- [ ] Adicionar o projeto à seção **Em destaque** do perfil — o post some do feed em dois dias, o
      perfil fica.
- [ ] Atualizar o título do perfil se fizer sentido.

## Ideias para os próximos posts

Consistência funciona melhor que um post isolado. Três ângulos que ainda não foram usados:

1. **Por que modelei o armazém como grafo** em vez de usar distância euclidiana — com a figura das
   rotas mostrando que o separador não atravessa as estruturas.
2. **Como o ACO funciona**, explicado sem jargão: formigas, feromônio, evaporação — usando a curva
   de convergência do `fig2`.
3. **O que o teste de escalabilidade revelou**: o ganho é mais previsível em ondas maiores, e por
   que isso importa para quem planeja a operação (`fig3`).
