# --*-- coding: utf-8 --*--
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

#Carregando a base de dados de treino e teste
treino = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')

#Array de colunas que nao existem no teste
colunasInexistentesTest = []

#Buscando as colunas que existem no treino mais nao existem no teste
for coluna in treino.columns:
    result = list(filter(lambda x:  x==coluna ,teste.columns))
    if not result:
        colunasInexistentesTest.append(coluna)

#Linha da nota de matematica
del(colunasInexistentesTest[69]) 

#Plotagem dos dados e suas correlacoes
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

#Apagando as colunas de treino que nao existem no teste e outras que nao tem correlacao com a target
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

#Remocao dos valores faltantes
#teste
row = list(teste.isnull().any(axis=1))
idxx = [i for i, j in enumerate(row) if j is True]
teste = teste.drop(idxx)

#Nvalores faltantes, substituindo por -1
imputer = SimpleImputer(strategy='constant', fill_value = -1)
treino = imputer.fit_transform(treino)
treino = pd.DataFrame(data = treino)

#Transformando colunas categoricas em numericas
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 12,13,14,15,16,17,18])], remainder='passthrough')                                       
treino = ct.fit_transform(treino)
treino = pd.DataFrame(data = treino)
treino.rename(columns = {54: "NU_MT"}, inplace=True)

ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 11,12,13,14,15,16,17])], remainder='passthrough')                                       
teste = ct2.fit_transform(teste)
teste = pd.DataFrame(data = teste)

#Parametrizacao
copia = treino['NU_MT']
treino["NU_NOTA_MT"] = copia
treino = treino.drop('NU_MT', axis=1)
base = treino
_, a = base.shape
atributos = base.iloc[:, 0:a - 1].values
classe = base.iloc[:, a - 1].values

#Faz o treinamento com regressao linear
regressor = LinearRegression()
regressor.fit(atributos, classe)

#Predicao do arquivo de teste
resultados = regressor.predict(teste)
