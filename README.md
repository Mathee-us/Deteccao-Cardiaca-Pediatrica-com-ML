# Uso de Modelos de Aprendizado de Máquina em Detecção de Patologias Cardíacas em Pacientes Pediátricos

## Descrição do Projeto

Este projeto tem como objetivo desenvolver e avaliar um sistema para a detecção precoce de patologias cardíacas em pacientes pediátricos, utilizando técnicas de aprendizado de máquina. Baseando-se em uma base de dados real coletada no Real Hospital Português (RHP), o estudo explora a relação entre diversas variáveis clínicas – como pressão arterial, idade, peso, altura, IMC e outros atributos – e a presença de doenças cardíacas. Essa abordagem visa auxiliar no diagnóstico precoce, contribuindo para intervenções terapêuticas mais eficazes e melhorando a qualidade de vida dos pacientes.

## Motivação

As doenças cardiovasculares em crianças e adolescentes frequentemente se manifestam de forma assintomática ou com sintomas inespecíficos, dificultando sua detecção sem exames especializados. Ao aplicar modelos de aprendizado de máquina, é possível analisar grandes volumes de dados clínicos de forma rápida e precisa, oferecendo uma ferramenta de apoio à decisão clínica e promovendo a prevenção de complicações.

## Conteúdo do Projeto

- **Dados:**
  - Registros médicos de pacientes pediátricos (e alguns adultos), contendo atributos como:
    - Peso, Altura, IMC (e faixa categorizada)
    - Pressão arterial sistólica, diastólica e pressão de pulso
    - Dados clínicos complementares (histórico de doenças, frequência cardíaca, etc.)

- **Pré-processamento:**
  - Tratamento de valores nulos e discrepantes
  - Remoção de atributos irrelevantes (ex.: datas, convênio)
  - Imputação de dados (utilizando métodos como Random Forest, KNN e Regressão Linear)
  - Tratamento de outliers (com base em boxplots e uso de mediana)
  - Conversão de atributos categóricos (one-hot encoding) e padronização das classes

- **Modelos e Experimentação:**
  - Avaliação de diversos algoritmos de classificação, incluindo:
    - Regressão Logística
    - Redes Neurais
    - SVM (Máquinas de Vetores de Suporte)
    - XGBoost
    - Random Forest
    - LightGBM
    - CatBoost
    - KNN e Naïve Bayes
  - Implementação de modelos de Stacking para combinar as forças de diferentes algoritmos, melhorando a robustez e a acurácia do sistema.
  - Otimização de hiperparâmetros por meio de GridSearch.

- **Resultados:**
  - Comparação de acurácia entre os modelos individuais e os modelos de Stacking.
  - Análise dos gráficos ROC para identificação de overfitting e avaliação da capacidade de generalização dos modelos.
  - Resultados demonstram alta acurácia na detecção de anomalias cardíacas, com os modelos de Stacking se destacando pela robustez.

## Requisitos e Dependências

- **Linguagem:** Python 3.x
- **Bibliotecas:**  
  - numpy  
  - pandas  
  - scikit-learn  
  - xgboost  
  - lightgbm  
  - catboost  
  - matplotlib  
  - seaborn  

## Conclusão
O projeto evidencia o potencial do uso de técnicas de aprendizado de máquina na área da saúde, oferecendo um suporte valioso para o diagnóstico precoce de anomalias cardíacas em pacientes pediátricos. As estratégias adotadas para o pré-processamento e a combinação de múltiplos modelos permitiram uma análise aprofundada dos dados clínicos, abrindo caminho para futuras investigações com conjuntos de dados ampliados e atributos adicionais.

## Referências
- G. Castro, "Sistema de suporte à decisão para a escolha do protocolo terapêutico para pacientes com leucemia mieloide aguda". Universidade Federal de São Carlos, 2023. Disponível em: Link
- S. Mohan, C. Thirumalai, and G. Srivastava, "Heart Disease Prediction Using Machine Learning Techniques", in 2019 IEEE 16th India Council In
