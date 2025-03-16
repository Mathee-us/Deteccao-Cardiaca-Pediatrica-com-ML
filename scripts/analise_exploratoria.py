# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: matheus rodrigues ferreira e Vinicius Fernandes Terra
# RA: 813919 e 814146
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria


#import das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def analisar_distribuicao(df, cols):
    """
    Funcao para analisar a distribuicao das classes no dataset

    df: DataFrame. Dataset a ser analisado
    """

    for col in cols:
        print(f"Distribuição de classes na coluna '{col}':")
        print(df[col].value_counts())
        print("\n")



def padronizar_colunas(df):
    """
    Funcao para padronizar o nome das colunas do dataset

    df: DataFrame. Dataset a ser padronizado
    """

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")
    df.rename(columns={'hda2': 'hda_2'}, inplace=True)
    df.rename(columns={'motivo1': 'motivo_1'}, inplace=True)
    df.rename(columns={'motivo2': 'motivo_2'}, inplace=True)


















