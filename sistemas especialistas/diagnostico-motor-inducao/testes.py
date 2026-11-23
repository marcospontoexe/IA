"""
Testes do sistema especialista do motor de indução trifásico.

Cada caso simula as leituras de uma situação conhecida e confere se o sistema
chega aos diagnósticos, às ações (na ordem de prioridade) e às hipóteses
esperados. Depois, confere se a validação recusa uma base e leituras com erro.
No fim, mostra quantas verificações passaram e se todas as regras da base
dispararam em pelo menos um caso (cobertura das regras).

Como rodar:
    python testes.py      (resumo dos casos)
    python testes.py -v   (mostra também o relatório completo de cada caso)
"""

import copy
import json
import os
import sys

from sistema_especialista import PASTA, carregar_base, diagnosticar, formatar_relatorio, validar_base

# Leituras de um motor saudável em operação. A resistência de isolamento fica de
# fora porque só pode ser medida com o motor parado.
NORMAL = {
    "vib_radial": 1.2, "vib_axial": 0.4,
    "pico_1x": False, "pico_2x": False, "freq_rolamento": False, "ruido_mancal": False,
    "temp_mancal": 62, "temp_enrolamento": 105, "corrente": 85,
    "deseq_tensao": 0.4, "deseq_corrente": 3,
}


def alterar(**mudancas):
    """Copia as leituras normais trocando só os valores indicados."""
    return {**NORMAL, **mudancas}


def sem(*nomes, **mudancas):
    """Copia as leituras normais sem as entradas indicadas (que ficam não informadas)."""
    leituras = alterar(**mudancas)
    for nome in nomes:
        del leituras[nome]
    return leituras


# O caso T11 usa o mesmo arquivo do exemplo do modo arquivo
with open(os.path.join(PASTA, "leitura_exemplo.json"), encoding="utf-8") as arquivo:
    EXEMPLO = json.load(arquivo)["leituras"]

# Cada caso: (nome, leituras, diagnósticos, ações em ordem de prioridade, regras das hipóteses)
CASOS = [
    ("T01 Motor em condição normal",
     NORMAL, [], [], []),
    ("T02 Rolamento com defeito inicial",
     alterar(freq_rolamento=True), ["D1"], ["A1"], []),
    ("T03 Rolamento com defeito avançado",
     alterar(freq_rolamento=True, ruido_mancal=True, temp_mancal=97), ["D1"], ["A2", "A1"], []),
    ("T04 Desbalanceamento severo",
     alterar(vib_radial=5.2, vib_axial=1.1, pico_1x=True), ["D2"], ["A9", "A3"], []),
    ("T05 Desalinhamento",
     alterar(vib_radial=3.6, vib_axial=2.6, pico_2x=True), ["D3"], ["A4"], []),
    ("T06 Sobrecarga",
     alterar(corrente=118, temp_enrolamento=146), ["D4"], ["A5"], []),
    ("T07 Falha no resfriamento",
     alterar(temp_enrolamento=148, corrente=93), ["D5"], ["A6"], []),
    ("T08 Alimentação desequilibrada",
     alterar(deseq_tensao=3.0, deseq_corrente=19, temp_enrolamento=146, corrente=97), ["D6"], ["A7"], []),
    ("T09 Curto entre espiras com o motor em operação",
     alterar(deseq_corrente=16, temp_enrolamento=125, corrente=98), ["D7"], ["A8"], []),
    ("T10 Isolamento baixo no ensaio com o motor parado",
     {"isolamento": 2.0}, ["D7"], ["A8"], []),
    ("T11 Falhas combinadas (leitura_exemplo.json)",
     EXEMPLO, ["D1", "D3"], ["A9", "A2", "A1", "A4"], []),
    ("T12 Só a vibração radial foi medida",
     {"vib_radial": 4.0}, [], [], ["R03", "R04", "R05"]),
    ("T13 Pico em 1× sem a medição axial",
     sem("vib_axial", vib_radial=3.4, pico_1x=True), [], [], ["R03", "R05"]),
]


def lista(itens, vazio="nenhum"):
    """Escreve uma lista de ids, ou o texto de vazio ("nenhum", "nenhuma") se ela estiver vazia."""
    return ", ".join(itens) if itens else vazio


def rodar_casos(base, detalhado):
    """Roda os casos de diagnóstico. Devolve o número de acertos e as regras disparadas."""
    acertos = 0
    regras_usadas = set()
    for nome, leituras, diag_esperados, acoes_esperadas, hip_esperadas in CASOS:
        resultado = diagnosticar(leituras, base)
        hipoteses = [regra["id"] for regra, _ in resultado["hipoteses"]]
        regras_usadas.update(regra["id"] for regra in resultado["disparadas"])

        passou = (resultado["diagnosticos"] == diag_esperados
                  and resultado["acoes"] == acoes_esperadas
                  and hipoteses == hip_esperadas)
        acertos += passou

        print(f"{'OK   ' if passou else 'FALHA'} {nome}")
        print(f"      diagnósticos: {lista(resultado['diagnosticos'])}"
              f" | ações: {lista(resultado['acoes'], 'nenhuma')}"
              f" | hipóteses: {lista(hipoteses, 'nenhuma')}")
        if not passou:
            print(f"      esperado: {lista(diag_esperados)} | {lista(acoes_esperadas, 'nenhuma')}"
                  f" | {lista(hip_esperadas, 'nenhuma')}")
        if detalhado:
            print(formatar_relatorio(base, resultado), end="\n\n")
    return acertos, regras_usadas


def recusa(funcao):
    """Diz se a função levanta ValueError, ou seja, se o erro foi detectado."""
    try:
        funcao()
    except ValueError:
        return True
    return False


def rodar_validacoes(base):
    """Confere se a base e as leituras com erro são recusadas. Devolve o número de acertos."""
    base_errada = copy.deepcopy(base)
    base_errada["regras"].append({"id": "R99", "se": ["S99"], "entao": "D1", "porque": "teste"})

    verificacoes = [
        ("V1 Regra com sintoma inexistente é recusada", lambda: validar_base(base_errada)),
        ("V2 Texto numa entrada numérica é recusado", lambda: diagnosticar({"vib_radial": "alta"}, base)),
        ("V3 Entrada com nome desconhecido é recusada", lambda: diagnosticar({"vibracao": 3.0}, base)),
    ]
    acertos = 0
    for nome, funcao in verificacoes:
        passou = recusa(funcao)
        acertos += passou
        print(f"{'OK   ' if passou else 'FALHA'} {nome}")
    return acertos, len(verificacoes)


def main():
    """Roda tudo, mostra o placar e termina com código 1 se alguma verificação falhar."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # acentos ao redirecionar a saída no Windows
    except (AttributeError, ValueError):
        pass

    base = carregar_base()
    acertos, regras_usadas = rodar_casos(base, detalhado="-v" in sys.argv)
    print()
    validas, total_validacoes = rodar_validacoes(base)

    todas = [regra["id"] for regra in base["regras"]]
    nao_usadas = [r for r in todas if r not in regras_usadas]
    print(f"\nCasos de diagnóstico: {acertos} de {len(CASOS)} passaram.")
    print(f"Validação da base e das leituras: {validas} de {total_validacoes} passaram.")
    print(f"Regras disparadas em pelo menos um caso: {len(todas) - len(nao_usadas)} de {len(todas)}"
          + (f" (faltam: {lista(nao_usadas)})." if nao_usadas else "."))

    if acertos < len(CASOS) or validas < total_validacoes or nao_usadas:
        sys.exit(1)


if __name__ == "__main__":
    main()
