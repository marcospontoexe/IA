# INTELIGENCIA ARTIFICIAL
 Este repositório contém alguns alguns projetos de inteligência artificial.
## JARBAS - Um assistente virtual por comando de voz para atomação residencial
O Jarbas foi um tcc desenvolvido durante meu curso de engenharia elétrica, concluido em 2018.
Neste trabalho foi desenvolvido um assistente virtual embarcado em uma placa de desenvolvimento **Orange Pi** com sistema operacional **Linux**. O assistente virtual consiste de um *tradutor homem/máquina* para comandos curtos de voz que, auxiliado por uma **Rede Neural Artificial** do tipo Multilayer Perceptron, que interpreta comandos de voz e envia a tradução destes para eletrodomésticos e equipamentos específicos através da rede Wi-Fi da residência pelo protocolo de comunicação **MQTT**. O assistente é ativado através do comando “Jarbas”, que é o nome que lhe foi atribuído. Depois de ativar o assistente, o usuário pode comandar os eletrodomésticos da casa através de comandos curtos como, por exemplo, “ligue tv” ou “desligue ar-condicionado”. Nesta primeira versão do assistente foram programados [dez comandos](https://github.com/marcospontoexe/IA/tree/main/Comandos%20de%20voz/Banco_de_palavras) (“Jarbas”, “Ligue”, “Desligue”, “Música”, “Sala”, “Quarto”, “Cozinha”, “Ar-condicionado”, “Tv” e “Café”). Para a construção desta base de dados de comandos, foram gravadas locuções dos dez comandos escolhidos por um grupo de 62 voluntários, incluindo homens, muheres e crianças. Além dos exemplos gravados, foram utilizadas técnicas de **Data Augmentation** para ampliar a base de dados com o intuito de aumentar a robustez da RNA treinada.
### Banco de dados
Para realizar a gravação dos voluntários, foi usado o código [Banco_Dados.py](https://github.com/marcospontoexe/IA/tree/main/Comandos%20de%20voz/JARBAS%20-%20Um%20assistente%20virtual%20por%20comando%20de%20voz%20para%20atoma%C3%A7%C3%A3o%20residencial/Banco%20de%20dados). Este código grava automaticamente três áudios de dois segundos para cada 
comando, com uma frequência de amostragem de 16000 bits por segundo. Os audios são salvos dentro da pasta em que se encontra o código "Banco_Dados.py", por tanto é necessário criar pastas distintas para 
cada comando gravado.


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
