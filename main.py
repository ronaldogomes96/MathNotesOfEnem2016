# --*-- coding: utf-8 --*--
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from yellowbrick.target.feature_correlation import feature_correlation
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
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
colunasNaoImportantes = ["NU_INSCRICAO","SG_UF_RESIDENCIA", "CO_PROVA_CN", "CO_PROVA_CH",
                         "CO_PROVA_LC", "CO_PROVA_MT", "Q027",  "CO_UF_RESIDENCIA", "NU_IDADE",
                         "TP_COR_RACA", "TP_NACIONALIDADE", "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU",
                         "TP_ESCOLA", "TP_ENSINO", "TP_DEPENDENCIA_ADM_ESC", "IN_BAIXA_VISAO", 
                         "IN_CEGUEIRA", "IN_SURDEZ", "IN_DISLEXIA", "IN_DISCALCULIA", "IN_SABATISTA",
                         "IN_GESTANTE","IN_IDOSO", "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC",
                         "TP_LINGUA", "TP_STATUS_REDACAO"]
for k in colunasNaoImportantes:
    treino = treino.drop(k, axis=1)
    teste = teste.drop(k, axis=1)
    

corr = treino.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(treino.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(treino.columns)
ax.set_yticklabels(treino.columns)
plt.show()


#Remocao dos valores faltantes
#treino
#rows = list(treino.isnull().any(axis=1))
#idx = [i for i, j in enumerate(rows) if j is True]
#treino = treino.drop(idx)
#teste
row = list(teste.isnull().any(axis=1))
idxx = [i for i, j in enumerate(row) if j is True]
teste = teste.drop(idxx)

#Nova abordagem para valores faltantes, substituindo por -1
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='constant', fill_value = -1)
#imputer.fit(treino)
treino = imputer.fit_transform(treino)
treino = pd.DataFrame(data = treino)
#treino.rename(columns = {24: "NU_NOTA_MT"}, inplace=True)

imputer2 = SimpleImputer(strategy='constant', fill_value = -1)
#imputer.fit(treino)
teste = imputer2.fit_transform(teste)
teste = pd.DataFrame(data = teste)




from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 12,13,14,15,16,17,18])], remainder='passthrough')                                       
treino = ct.fit_transform(treino)
treino = pd.DataFrame(data = treino)
treino.rename(columns = {54: "NU_MT"}, inplace=True)

ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 11,12,13,14,15,16,17])], remainder='passthrough')                                       
teste = ct2.fit_transform(teste)
teste = pd.DataFrame(data = teste)




"""
labelencoder = LabelEncoder()
colunasStrings = [0, 4, 24 ,25 ,26, 27, 40, 41, 42, 43, 44, 45, 46, 47]
treino[:, 0] = labelencoder.fit_transform(treino[:, 0])
"""
"""
for coluna in teste.select_dtypes(['object']).columns:
    teste[coluna] = pd.Categorical(teste[coluna], categories=teste[coluna].unique()).codes
"""


#Parametrizacao
copia = treino['NU_MT']
#treino.rename(columns = {54: "NU_NOTA_MT"}, inplace=True)
treino["NU_NOTA_MT"] = copia
treino = treino.drop('NU_MT', axis=1)


base = treino
_, a = base.shape
atributos = base.iloc[:, 0:a - 1].values
classe = base.iloc[:, a - 1].values




#Normalizacao
#scaler = StandardScaler()
#atributos = scaler.fit_transform(atributos)

#Divisao do dataset em treino e teste
atributos_treinamento, atributos_teste, classe_treinamento, classe_teste = train_test_split(atributos, classe, test_size=0.20, random_state=0)


#Faz o treinamento com regressao linear
regressor = LinearRegression()
#regressor = RandomForestRegressor(n_estimators = 10)
#regressor = MLPRegressor(hidden_layer_sizes = (9,9), max_iter = 10000000000000, activation = "logistic")

regressor.fit(atributos, classe)

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


