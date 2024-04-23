# INTELIGENCIA ARTIFICIAL
 Este repositório contém alguns alguns projetos de inteligência artificial.
## JARBAS - Um assistente virtual por comando de voz para atomação residencial
O Jarbas foi um tcc desenvolvido durante meu curso de engenharia elétrica, concluido em 2018.


Neste trabalho foi desenvolvido um assistente virtual embarcado em uma placa de desenvolvimento **Orange Pi** com sistema operacional **Linux**. O assistente virtual consiste de um *tradutor homem/máquina* para comandos curtos de voz que, auxiliado por uma **Rede Neural Artificial** do tipo Multilayer Perceptron, que interpreta comandos de voz e envia a tradução destes para eletrodomésticos e equipamentos específicos através da rede Wi-Fi da residência pelo protocolo de comunicação **MQTT**. 


O assistente é ativado através do comando “Jarbas”, que é o nome que lhe foi atribuído. Depois de ativar o assistente, o usuário pode comandar os eletrodomésticos da casa através de comandos curtos como, por exemplo, “ligue tv” ou “desligue ar-condicionado”. 


Nesta primeira versão do assistente foram programados [dez comandos](https://github.com/marcospontoexe/IA/tree/main/Comandos%20de%20voz/Banco_de_palavras) (“Jarbas”, “Ligue”, “Desligue”, “Música”, “Sala”, “Quarto”, “Cozinha”, “Ar-condicionado”, “Tv” e “Café”). Para a construção desta base de dados de comandos, foram gravadas locuções dos dez comandos escolhidos por um grupo de 62 voluntários, incluindo homens, muheres e crianças. Além dos exemplos gravados, foram utilizadas técnicas de **Data Augmentation** para ampliar a base de dados com o intuito de aumentar a robustez da RNA treinada.


### Banco de dados
Para realizar a gravação dos voluntários, foi usado o código [Banco_Dados.py](https://github.com/marcospontoexe/IA/tree/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/Banco%20de%20dados). Este código grava automaticamente três áudios de dois segundos para cada 
comando, com uma frequência de amostragem de 16000 bits por segundo. 


No terminal do linux digitar "arecord -l", para verificar o card e o device 
da entrada de som, para usar como parâmentro do comando "arecord".
Para fazer alterações no alsamixer, use o comando "sudo alsamixer" no terminal do linux,
para que as alterações sejam salvas após fechar o alsamixer. Se mesmo assim 
o alsamixer não estiver salvando as alterações após o sistema ser reiniciado, 
use o comando "sudo alsactl store" para gravar as alterações.
    

Caso esteja utilizando uma máquina virtual, certifique-se de que as entradas de
áudio estejam habilitadas e configuradas corretamente, de acordo com seu hardware.
Para teste de entrada de áudio, pode ser usado o audacity para realizar gravações
de teste.


É importante que ao final da gravação de voz, todos os audios sejam analisados para descartar as amostras inadequadas; aúdios cortados, aúdios inaudíveis e comandos incorretos. Ao termino da execução do código "Banco_Dados.py" serão gerados um total de 30 comandos (10 comandos x 3 repetições para cada comando). Os comandos devem ser armazenados em seus devidos diretórios, totalizando dez diretórios. Os comandos em cada diretório deveram ser renomeados com números inteiros positivo; 1, 2, 3, 4, 5 e assim por diante, para que possam ser lidos pelo código segmentador de audio, discutido a diante.


### Tratamento do banco de dados
A fim de reduzir a quantidade nos neurônios de entrada da RNA, tornando esta mais rápida e eficiente, os audios gravados com duração de dois segundos, que contém 32000 amostras, passam por algumas etapas de pré processamento; **pré-enfase**, **normalização**, **segmentação do audio**, filtro **MFCC** (*MEL FREQUENCY CEPSTRAL COEFICIENTE*). 


#### pré enfase
A pré-ênfase é usada para eliminar uma tendência espectral de aproximadamente -
6dB/oitava na fala, irradiada dos lábios. Essa compensação na atenuação pode ser aplicada
através de um **filtro FIR de primeira ordem**, de resposta aproximadamente +6dB/oitava, ocasionando um nivelamento no espectro. Enfatizando a porção do espectro mais distante da frequência
fundamental. A imagem a baixo mostra o comando de áudio "Jarbas"sem o filtro de pré-ênfase no
primeiro gráfico, e o áudio filtrado no segundo gráfico.

![Filtro de pré-enfase](https://github.com/marcospontoexe/IA/blob/main/Comandos%20de%20voz/imagens/pr%C3%A9%20enfase.png)


#### Normalização
A normalização é um processo em que o sinal de áudio passa por um ajuste de amplitude, fixando um valor de ganho máximo pretendido, neste caso entre -1 e 1, evitando problemas de convergência. Ter todas as características em uma escala similar pode evitar que algumas características dominem o processo de aprendizado. 


#### Segmentação do áudio
A segmentação do áudio é utilizada para selecionar as principais informações do sinal, separando o comando dito, de regiões de silêncio, que não contém informação alguma sobre o comando de áudio.
Para isso, foi desenvolvido um [algoritmo de segmentação](https://github.com/marcospontoexe/IA/tree/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/Tratamento%20do%20banco%20de%20dados%20e%20filtro%20mfcc) "segmentador.py" **baseado na energia contida no comando falado**, já que a região não falada possui apenas ruído de fundo, em que a energia é muito baixa.


A lógica por trás do algoritmo é detectar o início do comando a partir do limiar inferior
de energia, que separa a região de silencio da região de fala. O limiar inferior de energia é uma
proporção do pico de energia daquele comando de áudio. Para cada áudio analisado, o valor do limiar inferior de energia é alterado em função do pico de energia daquele áudio, dando dinâmica ao algoritmo de segmentação na determinação do início do comando falado. Em ambientes ruidosos basta que o usuário fale mais alto, assim o limiar inferior de energia também aumenta.


Após o áudio ser processador pelo **segmentador.py**, passou de 32000 para 17600 amostras. Para mais detalhes do código **segmentador.py** leia a seção 3.3.3 do pdf [JARBAS - Um assistente virtual por comando de voz para atomação residencial](https://github.com/marcospontoexe/IA/blob/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial.pdf).


A figura a baixo mostra a segmentação do comando "jarbas". No primeiro gráfico a energia ao longo do tempo, no segundo gráfico o áudio original, no terceiro gráfico a região não falada está zerada, no quarto gráfico o aúdio segmentado.

![Processo de segmentação do áudio](https://github.com/marcospontoexe/IA/blob/main/Comandos%20de%20voz/imagens/segmentador.png). 



### Filtro MFCC
Para alimentar uma RNA para reconhecimento de comandos vocálicos adequadamente, é necessário descartar componentes irrelevantes daquele padrão vocálico analisado, como por exemplo; ruído de fundo, emoção, gênero, entre outras características desnecessárias, tornando o processamento mais rápido e o reconhecimento mais eficiente. 


A extração usando MFCC é baseada em duas características da audição humana: a resposta em frequência da membrana basilar, onde existe uma banda crítica (leia a seção [2.4](https://github.com/marcospontoexe/IA/blob/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial.pdf)) é também a característica de excitações de compressão não lineares do nervo auditivo, geralmente é necessário 8 vezes mais energia para dobrar a intensidade de um sinal de áudio. Todo esse processamento é realizado pelo código [mfcc.py](https://github.com/marcospontoexe/IA/blob/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/Tratamento%20do%20banco%20de%20dados%20e%20filtro%20mfcc/mfcc.py)

Após o audio segmentado (com 17600 amostras) ser processado pelo código "mfcc.py", passou a ter apenas 130 amostras, que serão fornecidas à camada de entrada de RNA. Essas 130 amostras são as componentes cepstrais, que contém informação sobre a energia encontrada nas frequências que compõem aquele sinal de áudio.


Ao final do processo defiltragem MFCC, o código "mfcc.py" gera dois arquivos .CSV contendo todos as componentes cepstrais de cada comando. O arquivo chamado "treino.csv" será usado para treinar a RNA, e o arquivo "teste.csv" será usado para a validação da RNA.


Leia a seção [2.6 e 3.4](https://github.com/marcospontoexe/IA/blob/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial.pdf) para mais detalhes do algoritmo **MFCC**.

### Treino da RNA
* Em construção...*

### Jarbas
* Em construção...*
