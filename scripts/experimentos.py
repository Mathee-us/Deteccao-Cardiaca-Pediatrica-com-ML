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

# Arquivo com todas as funcoes e codigos referentes aos experimentos


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Função para testar combinações de features
def testar_combinacoes_features(x_train, y_train, grupos_excluidos, param_grid):
    # Achatando a lista de grupos para obter as colunas a serem excluídas
    cols_excluidas = [col for grupo in grupos_excluidos for col in grupo]
    x_train_reduzido = x_train.drop(columns=cols_excluidas)

    # Configurar o GridSearchCV para otimização
    grid_search = GridSearchCV(
        estimator=XGBClassifier(eval_metric='logloss', tree_method="hist", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(x_train_reduzido, y_train)

    # Obter o melhor modelo após a otimização
    melhor_modelo = grid_search.best_estimator_

    # Avaliar o modelo no conjunto de treino
    y_train_pred = melhor_modelo.predict(x_train_reduzido)
    acuracia = accuracy_score(y_train, y_train_pred)

    return {str(grupos_excluidos): acuracia}









