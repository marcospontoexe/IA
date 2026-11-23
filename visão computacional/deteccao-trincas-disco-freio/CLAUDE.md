# CLAUDE.md

Este arquivo orienta o Claude Code ao trabalhar nesta pasta.

## O que é

Pipeline em Python e OpenCV que detecta trincas térmicas num disco de freio a partir de uma foto, separa-as dos riscos de desgaste e decide se o disco passa na triagem. É um projeto de portfólio: o [README.md](README.md) é a documentação para quem visita o repositório. Escreva sempre em pt-BR, e mantenha o tom de projeto (sem referências a curso, atividade ou entrega).

## Arquivos

- [disco_freio_trincado.jpg](disco_freio_trincado.jpg): disco de freio ventilado com trincas térmicas.
  - É a versão de 1280 × 877 px de um arquivo do Wikimedia Commons, de Johann H. Addicks, editado por FloSch, com licença CC BY-SA 3.0.
  - O Commons recusa download automático (erro 429); a foto foi baixada pelo navegador.
  - As imagens derivadas seguem a mesma licença, e o README dá os créditos. Não remova os créditos.
- [deteccao_trincas.py](deteccao_trincas.py): o pipeline de 9 etapas (cinza, equalização, gaussiano, pista desenrolada, média de 1 × 5 ao longo do raio, Sobel de 2ª ordem na volta, limiar fixo de 75, classificação por forma e resultado na foto). Grava 14 imagens em [imagens/](imagens/).
- [requirements.txt](requirements.txt): OpenCV e NumPy, sem versão fixa. O código foi testado com Python 3.13, OpenCV 5.0 e NumPy 2.3.

## Comandos

Nesta máquina, o `cv2` só está no Python do Anaconda (`$env:USERPROFILE\anaconda3\python.exe`). Nem ele nem o `conda` estão no PATH.

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -B deteccao_trincas.py           # analisa a foto e grava imagens/
& "$env:USERPROFILE\anaconda3\python.exe" -B deteccao_trincas.py --teste   # disco sintético com 8 trincas
```

- O `--teste` grava numa pasta temporária e termina com código 1 se não achar 2 trincas por quadrante.
- O código foi testado no Spyder 6.1, com F5 e com `runfile('deteccao_trincas.py', args='--teste')` no console. No Spyder, um `sys.exit(0)` aparece como "SystemExit: 0", com cara de erro; por isso o código só chama `sys.exit(1)` quando o teste falha.
- Para rodar um script no console do Spyder por fora: os kernels ficam em `%APPDATA%\jupyter\runtime\kernel-*.json`, e o do console é o que está com o diretório de trabalho na pasta do script. Use o `jupyter_client` com `runfile(...)`. O console mostra os erros vindos de outros clientes, mas não a saída de texto deles.

## Caminhos com acento

O `cv2.imread` e o `cv2.imwrite` não abrem caminhos com acento no Windows, e o caminho deste projeto tem "visão". O código lê e grava com `np.fromfile` + `cv2.imdecode` e `cv2.imencode` + `tofile`. Faça o mesmo em qualquer código novo com OpenCV aqui.

## Decisões que custaram iterações

- Na foto oblíqua, a face do disco é uma elipse. O centro e o semieixo horizontais vêm dos extremos da linha mais larga da silhueta do Otsu; o centro e o semieixo verticais, de um ajuste por mínimos quadrados ao arco de cima. Usar só a "primeira linha mais larga" põe o centro uns 20 px acima do real.
- As linhas da faixa desenrolada são espaçadas por comprimento de arco. Com ângulos iguais, as laterais da elipse ficam esticadas, e as trincas ali somem.
- A direção de cada mancha vem do eixo principal (covariância dos pixels), com 30° de tolerância.
- O limiar é fixo porque precisa ser absoluto; nesta foto, o Otsu daria 71, quase igual. O percentil marcaria sempre a mesma fração, mesmo num disco sem defeito.
- As versões descartadas estão na seção "Decisões de projeto" do README: laplaciano com direção por mancha (achou 1 das 8 trincas do sintético) e faixa com ângulos iguais (achou 6).

## Manter o README sincronizado

O [README.md](README.md) cita números da execução: 81 trincas (8, 33, 19 e 21 por quadrante), 276 manchas descartadas, elipse (626, 372) com semieixos 609 × 347, 13.784 de 315.960 pixels, o trecho ampliado em (0, 286) e os resultados do teste sintético. Se o código ou os parâmetros mudarem, rode os dois comandos de novo e atualize as seções "Resultados", "Como funciona, etapa por etapa" e "Validação com um disco sintético".
