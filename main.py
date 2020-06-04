# --*-- coding: utf-8 --*--

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv


#Carregando a base de dados de treino e teste
treino = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')
test = teste

#Array de colunas que nao existem no teste
colunasInexistentesTest = []

#Buscando as colunas que existem no treino mais nao existem no teste
for coluna in treino.columns:
    result = list(filter(lambda x:  x==coluna ,teste.columns))
    if not result:
        colunasInexistentesTest.append(coluna)

#Linha da nota de matematica
del(colunasInexistentesTest[69]) 

#Apagando as colunas de treino que nao existem no teste
treino = treino.drop(colunasInexistentesTest, axis=1)

#Remocao dos valores faltantes
#treino
rows = list(treino.isnull().any(axis=1))
idx = [i for i, j in enumerate(rows) if j is True]
treino = treino.drop(idx)
#teste
row = list(teste.isnull().any(axis=1))
idxx = [i for i, j in enumerate(row) if j is True]
teste = teste.drop(idxx)


# Transformando as colunas categóricas em numéricas
for coluna in treino.select_dtypes(['object']).columns:
    treino[coluna] = pd.Categorical(treino[coluna], categories=treino[coluna].unique()).codes
    
for coluna in teste.select_dtypes(['object']).columns:
    teste[coluna] = pd.Categorical(teste[coluna], categories=teste[coluna].unique()).codes

#Parametrizacao
classe = ['NU_NOTA_MT']
coluna = treino.columns[:-len(classe)]
treino = treino.reindex(columns=list(coluna) + classe)
base = treino
_, a = base.shape
atributos = base.iloc[:, 0:a - 1].values
classe = base.iloc[:, a - 1].values

#Normalizacao
scaler = StandardScaler()
atributos = scaler.fit_transform(atributos)

#Divisao do dataset em treino e teste
atributos_treinamento, atributos_teste, classe_treinamento, classe_teste = train_test_split(atributos, classe, test_size=0.20, random_state=0)


#Faz o treinamento com regressao linear
#regressor = LinearRegression()
#regressor = RandomForestRegressor(n_estimators = 10)
regressor = MLPRegressor(hidden_layer_sizes = (9,9), max_iter = 10000000000000, activation = "logistic")

regressor.fit(atributos_treinamento, classe_treinamento)

#Score da relação enrte x e y
score = regressor.score(atributos_treinamento, classe_treinamento)

#Faz o treinamento
previsoes = regressor.predict(atributos_teste)

#Mostra a diferença entre o valor real e o valor treinado
mae = mean_absolute_error(classe_teste, previsoes)
mse = mean_squared_error(classe_teste, previsoes)

#Resultados
inscricao = []
for i in test.index:
    if row[i] == False:
        inscricao.append(test["NU_INSCRICAO"][i])
resultados = regressor.predict(teste)
print(test["NU_INSCRICAO"])


