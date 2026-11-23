"""
Sistema especialista para diagnóstico e manutenção de um motor de indução trifásico.

Diagnostica falhas a partir de leituras de sensores e de observações da manutenção,
explica cada conclusão e recomenda ações em ordem de prioridade. Documentação no README.md.

O programa segue a estrutura clássica de um sistema especialista:

- Base de conhecimento: o arquivo base_conhecimento.json, separado do código.
  Ele guarda as entradas medidas, os sintomas (com os limites), os diagnósticos,
  as ações de manutenção e as regras SE-ENTÃO.
- Motor de inferência: a função inferir(), com encadeamento para frente.
- Módulo de explicação: cada conclusão guarda a regra que a gerou e o porquê.
- Interface: perguntas no terminal ou leitura de um arquivo JSON com as medições,
  no formato que um sistema de monitoramento poderia enviar.
- Aquisição de conhecimento: o especialista edita o JSON, e validar_base()
  confere a base sempre que ela é carregada.

Como rodar:
    python sistema_especialista.py                       (modo interativo)
    python sistema_especialista.py leitura_exemplo.json  (lê as medições de um arquivo)

Só usa a biblioteca padrão do Python 3.
"""

import json
import operator
import os
import sys
import textwrap

# Pasta deste arquivo: a base é procurada aqui, e não na pasta de onde o script foi chamado
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_BASE = os.path.join(PASTA, "base_conhecimento.json")

# Comparações que a base pode usar para transformar uma leitura num sintoma
OPERADORES = {">": operator.gt, "<": operator.lt, "==": operator.eq}

# Prioridades das ações de manutenção, da mais grave para a menos grave
PRIORIDADES = ["imediata", "urgente", "programada"]

# Prefixo que nega uma condição de regra: "NAO S03" quer dizer "o sintoma S03 está ausente"
NEGACAO = "NAO "


# ---------------------------------------------------------------------------
# Base de conhecimento
# ---------------------------------------------------------------------------

def carregar_base(caminho=ARQUIVO_BASE):
    """Lê a base de conhecimento do arquivo JSON e confere se ela está consistente."""
    with open(caminho, encoding="utf-8") as arquivo:
        base = json.load(arquivo)
    validar_base(base)
    return base


def separar_negacao(condicao):
    """Divide uma condição de regra: "NAO S03" vira (True, "S03"), e "S01" vira (False, "S01")."""
    if condicao.startswith(NEGACAO):
        return True, condicao[len(NEGACAO):]
    return False, condicao


def validar_base(base):
    """Procura erros na base, como uma regra que cita um sintoma inexistente.

    Assim, um erro de digitação de quem editou o JSON aparece ao carregar a base,
    e não no meio de um diagnóstico. Levanta ValueError com todos os erros achados.
    """
    erros = []
    entradas, sintomas = base["entradas"], base["sintomas"]
    diagnosticos, acoes = base["diagnosticos"], base["acoes"]

    # Cada sintoma precisa apontar para entradas existentes e usar um operador conhecido
    for sid, sintoma in sintomas.items():
        for chave in ("entrada", "relativo_a"):
            if chave in sintoma and sintoma[chave] not in entradas:
                erros.append(f"{sid}: a entrada '{sintoma[chave]}' não existe")
        if sintoma["operador"] not in OPERADORES:
            erros.append(f"{sid}: operador '{sintoma['operador']}' desconhecido")

    # Cada ação precisa de uma das prioridades conhecidas
    for aid, acao in acoes.items():
        if acao["prioridade"] not in PRIORIDADES:
            erros.append(f"{aid}: prioridade '{acao['prioridade']}' desconhecida")

    # Cada regra precisa de um id único, de condições conhecidas e de uma conclusão válida
    ids = set()
    conhecidos = set(sintomas) | set(diagnosticos) | set(acoes)
    for regra in base["regras"]:
        if regra["id"] in ids:
            erros.append(f"{regra['id']}: id repetido")
        ids.add(regra["id"])
        for condicao in regra["se"]:
            negada, nome = separar_negacao(condicao)
            if nome not in conhecidos:
                erros.append(f"{regra['id']}: a condição '{nome}' não existe")
            elif negada and nome not in sintomas:
                # Só sintomas podem ser negados, porque são medidos. Um diagnóstico
                # ainda não concluído não quer dizer que ele foi descartado.
                erros.append(f"{regra['id']}: só sintomas podem ser negados ('{condicao}')")
        if regra["entao"] not in diagnosticos and regra["entao"] not in acoes:
            erros.append(f"{regra['id']}: a conclusão '{regra['entao']}' não é diagnóstico nem ação")

    if erros:
        raise ValueError("Base de conhecimento inválida:\n  " + "\n  ".join(erros))


def validar_leituras(base, leituras):
    """Confere se cada leitura tem um nome conhecido e o tipo certo (número ou sim/não)."""
    erros = []
    for nome, valor in leituras.items():
        entrada = base["entradas"].get(nome)
        if entrada is None:
            erros.append(f"entrada desconhecida: '{nome}'")
        elif entrada["tipo"] == "sim_nao" and not isinstance(valor, bool):
            erros.append(f"'{nome}' precisa ser true ou false")
        elif entrada["tipo"] == "numero" and (isinstance(valor, bool) or not isinstance(valor, (int, float))):
            # Em Python, bool é um tipo de int, então True passaria como número sem este teste
            erros.append(f"'{nome}' precisa ser um número")
    if erros:
        raise ValueError("Leituras inválidas:\n  " + "\n  ".join(erros))


# ---------------------------------------------------------------------------
# Motor de inferência
# ---------------------------------------------------------------------------

def avaliar_sintomas(base, leituras):
    """Transforma as leituras em sintomas, cada um com um de três valores.

    True quer dizer presente, False quer dizer ausente e None quer dizer desconhecido.
    Um sintoma é desconhecido quando falta alguma leitura de que ele depende.
    """
    estado = {}
    for sid, sintoma in base["sintomas"].items():
        medido = leituras.get(sintoma["entrada"])
        limite = sintoma["valor"]
        if "relativo_a" in sintoma:
            # Limite que depende de outra leitura (S03: vib_axial > 0,5 × vib_radial)
            referencia = leituras.get(sintoma["relativo_a"])
            limite = None if referencia is None else limite * referencia
        if medido is None or limite is None:
            estado[sid] = None
        else:
            estado[sid] = OPERADORES[sintoma["operador"]](medido, limite)
    return estado


def valor_condicao(condicao, fatos):
    """Valor de uma condição de regra: True, False ou None (desconhecido)."""
    negada, nome = separar_negacao(condicao)
    # Diagnósticos e ações ainda não concluídos contam como falsos
    valor = fatos.get(nome, False)
    if valor is None:
        return None  # a negação de algo desconhecido continua desconhecida
    return not valor if negada else valor


def avaliar_regra(regra, fatos):
    """Faz o E lógico das condições de uma regra, com três valores.

    O resultado é False se alguma condição for falsa. É None se nenhuma for falsa,
    mas alguma for desconhecida. É True só quando todas são verdadeiras.
    """
    valores = [valor_condicao(c, fatos) for c in regra["se"]]
    if False in valores:
        return False
    if None in valores:
        return None
    return True


def inferir(base, sintomas):
    """Motor de inferência com encadeamento para frente (forward chaining).

    Parte dos sintomas e dispara toda regra cujas condições são verdadeiras. Cada
    conclusão vira um fato novo, que pode disparar outras regras: um diagnóstico
    dispara a ação de manutenção dele. O laço termina quando uma volta inteira
    pelas regras passa sem nenhum disparo novo.

    Depois, procura hipóteses: regras de diagnóstico que não dispararam só porque
    faltam dados. É a ideia do encadeamento para trás, em que o sistema indica o
    que precisa ser medido para confirmar um diagnóstico.

    Devolve os fatos, as regras disparadas, na ordem do disparo, e as hipóteses.
    """
    fatos = dict(sintomas)
    disparadas = []
    ids_disparados = set()

    mudou = True
    while mudou:
        mudou = False
        for regra in base["regras"]:
            if regra["id"] not in ids_disparados and avaliar_regra(regra, fatos) is True:
                fatos[regra["entao"]] = True  # a conclusão vira um fato novo
                ids_disparados.add(regra["id"])
                disparadas.append(regra)      # guardada para o módulo de explicação
                mudou = True

    # Uma hipótese precisa de pelo menos uma condição verdadeira. Sem essa exigência,
    # todo diagnóstico viraria hipótese quando quase nada foi medido.
    hipoteses = []
    for regra in base["regras"]:
        conclusao = regra["entao"]
        if conclusao in base["diagnosticos"] and conclusao not in fatos:
            valores = [valor_condicao(c, fatos) for c in regra["se"]]
            if avaliar_regra(regra, fatos) is None and True in valores:
                faltando = [c for c, v in zip(regra["se"], valores) if v is None]
                hipoteses.append((regra, faltando))

    return fatos, disparadas, hipoteses


def diagnosticar(leituras, base=None):
    """Faz uma consulta completa: valida as leituras, avalia os sintomas e roda o motor.

    Uma leitura com valor None conta como não informada. Devolve um dicionário
    com tudo o que a interface e os testes precisam.
    """
    if base is None:
        base = carregar_base()
    leituras = {nome: valor for nome, valor in leituras.items() if valor is not None}
    validar_leituras(base, leituras)

    sintomas = avaliar_sintomas(base, leituras)
    fatos, disparadas, hipoteses = inferir(base, sintomas)

    # Diagnósticos na ordem da base; ações da mais grave para a menos grave
    # (o sorted é estável, então ações de mesma prioridade mantêm a ordem da base)
    diagnosticos = [d for d in base["diagnosticos"] if fatos.get(d)]
    acoes = sorted((a for a in base["acoes"] if fatos.get(a)),
                   key=lambda a: PRIORIDADES.index(base["acoes"][a]["prioridade"]))

    return {"leituras": leituras, "sintomas": sintomas, "disparadas": disparadas,
            "diagnosticos": diagnosticos, "acoes": acoes, "hipoteses": hipoteses}


# ---------------------------------------------------------------------------
# Interface e módulo de explicação
# ---------------------------------------------------------------------------

def numero(valor):
    """Formata um número no padrão brasileiro, com vírgula decimal (2.8 vira "2,8")."""
    return f"{valor:g}".replace(".", ",")


def quebrar(texto, recuo=6, recuo_seguinte=None):
    """Quebra um texto longo em linhas de até 78 caracteres, com recuo à esquerda.

    recuo_seguinte é o recuo da segunda linha em diante (por padrão, igual ao da primeira).
    """
    seguinte = recuo if recuo_seguinte is None else recuo_seguinte
    return textwrap.fill(texto, width=78, initial_indent=" " * recuo, subsequent_indent=" " * seguinte)


def regra_em_texto(regra):
    """Escreve uma regra no formato SE-ENTÃO, como "R03 (SE S01 E S04 E NÃO S03 ENTÃO D2)"."""
    condicoes = " E ".join(c.replace(NEGACAO, "NÃO ") for c in regra["se"])
    return f"{regra['id']} (SE {condicoes} ENTÃO {regra['entao']})"


def formatar_relatorio(base, resultado, cabecalho=()):
    """Monta o texto que o operador lê: sintomas, diagnósticos, ações e hipóteses."""
    leituras = resultado["leituras"]
    linhas = ["=" * 78, f"Diagnóstico: {base['equipamento']['nome']}", *cabecalho, "=" * 78]

    # 1) Sintomas presentes, com a leitura que ativou cada um
    linhas.append("\nSintomas presentes:")
    presentes = [sid for sid, valor in resultado["sintomas"].items() if valor]
    for sid in presentes:
        sintoma = base["sintomas"][sid]
        medido = leituras[sintoma["entrada"]]
        detalhe = ""
        if not isinstance(medido, bool):
            unidade = base["entradas"][sintoma["entrada"]]["unidade"]
            detalhe = f" (lido: {numero(medido)} {unidade}"
            if "relativo_a" in sintoma:
                # Mostra a conta do limite relativo, como 0,5 × 4,8 = 2,4
                ref = leituras[sintoma["relativo_a"]]
                limite = sintoma["valor"] * ref
                detalhe += f"; limite: {numero(sintoma['valor'])} × {numero(ref)} = {numero(round(limite, 2))} {unidade}"
            detalhe += ")"
        linhas.append(quebrar(f"{sid}  {sintoma['descricao']}{detalhe}", recuo=2, recuo_seguinte=7))
    if not presentes:
        linhas.append("  nenhum")

    # 2) Diagnósticos com o porquê: é aqui que entra o módulo de explicação
    linhas.append("\nDiagnósticos:")
    for d in resultado["diagnosticos"]:
        linhas.append(f"  {d}  {base['diagnosticos'][d]['descricao']}")
        for regra in resultado["disparadas"]:
            if regra["entao"] == d:
                linhas.append(f"      Por quê: {regra_em_texto(regra)}")
                linhas.append(quebrar(regra["porque"], recuo=10))
    if not resultado["diagnosticos"]:
        linhas.append("  nenhum confirmado")

    # 3) Ações de manutenção, da mais grave para a menos grave, com a regra de origem
    linhas.append("\nRecomendações de manutenção (da mais grave para a menos grave):")
    for a in resultado["acoes"]:
        acao = base["acoes"][a]
        linhas.append(f"  [{acao['prioridade'].upper()}] {a}")
        linhas.append(quebrar(acao["descricao"]))
        origens = [regra_em_texto(r) for r in resultado["disparadas"] if r["entao"] == a]
        linhas.append(f"      Origem: {'; '.join(origens)}")
    if not resultado["acoes"]:
        if resultado["hipoteses"]:
            linhas.append("  nenhuma confirmada; colete os dados pedidos nas hipóteses abaixo")
        else:
            linhas.append("  nenhuma")

    # 4) Hipóteses: o que falta medir para confirmar ou descartar um diagnóstico
    if resultado["hipoteses"]:
        linhas.append("\nHipóteses a confirmar (faltam dados):")
        for regra, faltando in resultado["hipoteses"]:
            d = regra["entao"]
            linhas.append(f"  {d}  {base['diagnosticos'][d]['descricao']}, pela regra {regra['id']}, se:")
            for condicao in faltando:
                negada, sid = separar_negacao(condicao)
                sintoma = base["sintomas"][sid]
                estado = "ausente" if negada else "presente"
                # Lista só as entradas que ainda não foram informadas
                necessarias = [sintoma["entrada"], sintoma.get("relativo_a")]
                informar = ", ".join(e for e in necessarias if e and e not in leituras)
                texto = f"{sid} estiver {estado}: {sintoma['descricao']} (informe {informar})"
                linhas.append(quebrar(texto, recuo=6, recuo_seguinte=10))

    # 5) Conclusão geral e entradas que ficaram sem leitura
    if not (resultado["diagnosticos"] or resultado["acoes"] or resultado["hipoteses"]):
        linhas.append("\nNenhuma falha identificada com os dados informados. Manter o monitoramento de rotina.")
    faltam = [nome for nome in base["entradas"] if nome not in leituras]
    if faltam:
        linhas.append("\n" + quebrar(f"Entradas não informadas: {', '.join(faltam)}", recuo=0, recuo_seguinte=2))

    return "\n".join(linhas)


def perguntar_leituras(base):
    """Interface de terminal: pergunta cada entrada da base; Enter vazio = não informado."""
    print("Informe as leituras do motor. Deixe em branco (Enter) o que não foi medido.\n")
    leituras = {}
    for nome, entrada in base["entradas"].items():
        sufixo = " (s/n)" if entrada["tipo"] == "sim_nao" else ""
        while True:
            resposta = input(f"{entrada['pergunta']}{sufixo}: ").strip().lower()
            if resposta == "":
                break  # não informado: o sintoma que depende desta entrada fica desconhecido
            if entrada["tipo"] == "numero":
                try:
                    leituras[nome] = float(resposta.replace(",", "."))  # aceita 2,8 e 2.8
                    break
                except ValueError:
                    print("  Digite um número, como 2,8.")
            elif resposta in ("s", "sim"):
                leituras[nome] = True
                break
            elif resposta in ("n", "nao", "não"):
                leituras[nome] = False
                break
            else:
                print("  Responda s ou n.")
    return leituras


def main():
    """Ponto de entrada: lê as leituras (de um arquivo ou do teclado) e mostra o diagnóstico."""
    # Garante acentos e símbolos (Ω, ×) quando a saída é redirecionada para um arquivo
    # no Windows. No Spyder, a saída não tem reconfigure() e já aceita esses caracteres.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    base = carregar_base()
    if len(sys.argv) > 1:
        # Modo arquivo: as leituras chegam num JSON, como as enviaria um sistema de monitoramento
        with open(sys.argv[1], encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        leituras = dados["leituras"]
        cabecalho = [f"{rotulo}: {dados[chave]}"
                     for chave, rotulo in (("equipamento", "Equipamento"), ("data_hora", "Leitura"))
                     if chave in dados]
    else:
        leituras = perguntar_leituras(base)
        cabecalho = []
    print(formatar_relatorio(base, diagnosticar(leituras, base), cabecalho))


if __name__ == "__main__":
    main()
