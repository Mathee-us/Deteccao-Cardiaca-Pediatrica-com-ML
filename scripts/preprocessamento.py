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

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento
#import das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression


#todo: importar como pre
#removendo outliers, testar depois substituindo os outliers
def remove_outliers(df, col):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] <= (Q3 + 1.5 * IQR)) & (df[col] >= (Q1 - 1.5 * IQR))]
    return df

def plot_train_vs_test(data_train,data_test, col):
    fig, ax = plt.subplots(figsize=(18, 5))

    sns.kdeplot(data_train[col], color='blue', label='Train ', alpha=0.7, ax=ax)
    sns.kdeplot(data_test[col], color='orange', label='Test ', alpha=0.7, ax=ax)

    ax.set_title('Treino x Teste')
    ax.set_xlabel(col)
    ax.set_ylabel('Densidade')
    ax.legend()

    plt.show()

def replace_outliers_with_median(df, col):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    median = df[col].median()

    df[col] = df[col].apply(lambda x: median if x > (Q3 + 1.5 * IQR) else x)
    df[col] = df[col].apply(lambda x: median if x < (Q1 - 1.5 * IQR) else x)

    return df

#substituindo valores 0 de altura pelo modelo de random forest
#todo: implementar essa função
def preencher_alturas_randomForest(data):
    # Filtrar dados com altura conhecida e desconhecida
    df_treino = data[data['altura'] > 0].copy()
    df_teste = data[data['altura'] == 0].copy()

    # Criar variáveis dummy para 'sexo' (caso seja categórico)
    df_treino = pd.get_dummies(df_treino, columns=['sexo'], drop_first=True)
    df_teste = pd.get_dummies(df_teste, columns=['sexo'], drop_first=True)

    # Garantir que ambos os conjuntos têm as mesmas colunas
    df_teste = df_teste.reindex(columns=df_treino.columns, fill_value=0)

    # Atualizar lista de features após dummies
    features = [col for col in df_treino.columns if col not in ['altura']]

    # Dividir dados de treino
    X_train = df_treino[features]
    y_train = df_treino['altura']

    # Treinar modelode regressão linear
    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)

    # Prever alturas ausentes
    X_test = df_teste[features]
    data.loc[data['altura'] == 0, 'altura'] = modelo.predict(X_test)

    return data

def preencher_altura_knn(df):
    df = df.copy()

    # Converte a coluna 'sexo' para valores numéricos
    label_encoder = LabelEncoder()
    df['sexo'] = label_encoder.fit_transform(df['sexo'])

    # Seleciona as colunas para imputação
    colunas_treino = ['idade', 'sexo', 'peso', 'altura']

    # Substituir valores 0 por NaN para que o KNNImputer possa trabalhar corretamente
    df['altura'] = df['altura'].replace(0, np.nan)

    # Criar e aplicar o imputador KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    df[colunas_treino] = knn_imputer.fit_transform(df[colunas_treino])

    return df

def preencher_altura_regressao_linear(data):
    data = data.copy()

    # Filtrar dados com altura conhecida e desconhecida
    df_treino = data[data['altura'] > 0].copy()
    df_teste = data[data['altura'] == 0].copy()

    # Criar variáveis dummy para 'sexo' (caso seja categórico)
    df_treino = pd.get_dummies(df_treino, columns=['sexo'], drop_first=True)
    df_teste = pd.get_dummies(df_teste, columns=['sexo'], drop_first=True)

    # Garantir que ambos os conjuntos têm as mesmas colunas
    df_teste = df_teste.reindex(columns=df_treino.columns, fill_value=0)

    # Atualizar lista de features após dummies
    features = [col for col in df_treino.columns if col not in ['altura']]

    # Verificar se há dados suficientes para treinar o modelo
    if df_treino.empty or df_teste.empty:
        print("Não há dados suficientes para realizar a imputação.")
        return data

    # Dividir dados de treino
    X_train = df_treino[features]
    y_train = df_treino['altura']

    # Treinar modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Prever alturas ausentes
    X_test = df_teste[features]
    data.loc[data['altura'] == 0, 'altura'] = modelo.predict(X_test)

    return data


#substituindo os valores de peso=0 pelo modelo knn
def preencher_peso_knn(df):
    df = df.copy()

    # Converte a coluna 'sexo' para valores numéricos
    label_encoder = LabelEncoder()
    df['sexo'] = label_encoder.fit_transform(df['sexo'])

    # Seleciona as colunas para imputação
    colunas_treino = ['idade', 'sexo', 'altura', 'peso']

    # Substituir valores 0 por NaN para que o KNNImputer possa trabalhar corretamente
    df['peso'] = df['peso'].replace(0, np.nan)

    # Criar e aplicar o imputador KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    df[colunas_treino] = knn_imputer.fit_transform(df[colunas_treino])

    return df


def preencher_peso_regressao_linear(data):
    data = data.copy()

    # Selecionar apenas as colunas relevantes
    colunas_usadas = ['idade', 'sexo', 'altura', 'peso']
    data = data[colunas_usadas]

    # Filtrar dados com peso conhecido e desconhecido
    df_treino = data[data['peso'] > 0].copy()  # Dados com peso conhecido
    df_teste = data[data['peso'] == 0].copy()  # Dados com peso desconhecido

    # Criar variáveis dummy para 'sexo' (caso seja categórico)
    df_treino = pd.get_dummies(df_treino, columns=['sexo'], drop_first=True)
    df_teste = pd.get_dummies(df_teste, columns=['sexo'], drop_first=True)

    # Garantir que ambos os conjuntos têm as mesmas colunas
    df_teste = df_teste.reindex(columns=df_treino.columns, fill_value=0)

    # Atualizar lista de features após dummies
    features = [col for col in df_treino.columns if col not in ['peso']]

    # Dividir dados de treino
    X_train = df_treino[features]
    y_train = df_treino['peso']

    # Treinar modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Prever pesos ausentes
    X_test = df_teste[features]
    data.loc[data['peso'] == 0, 'peso'] = modelo.predict(X_test)

    return data

# testando random forest para preencher os valores de peso=0
def preencher_peso_randomForest(data):
    # Filtrar dados com peso conhecido e desconhecido
    df_treino = data[data['peso'] > 0].copy()
    df_teste = data[data['peso'] == 0].copy()

    # Criar variáveis dummy para 'sexo' (caso seja categórico)
    df_treino = pd.get_dummies(df_treino, columns=['sexo'], drop_first=True)
    df_teste = pd.get_dummies(df_teste, columns=['sexo'], drop_first=True)

    # Garantir que ambos os conjuntos têm as mesmas colunas
    df_teste = df_teste.reindex(columns=df_treino.columns, fill_value=0)

    # Atualizar lista de features após dummies
    features = [col for col in df_treino.columns if col not in ['peso']]

    # Dividir dados de treino
    X_train = df_treino[features]
    y_train = df_treino['peso']

    # Treinar modelo de regressão linear
    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)

    # Prever pesos ausentes
    X_test = df_teste[features]
    data.loc[data['peso'] == 0, 'peso'] = modelo.predict(X_test)

    return data

# recalcular o imc
def recalcular_imc(df):
    df = df.copy()

    df['imc'] = df['peso'] / ((df['altura'] / 100) ** 2)

    return df

# criando uma coluna categorica com faixa de imcs
def criar_faixa_imc(df):
    df = df.copy()

    df['faixa_imc'] = pd.cut(df['imc'], bins=[0, 10, 12.5, 15, 17.5, 30], labels=['0-10', '10-12.5', '12.5-15', '15-17.5', '17.5>'])

    return df