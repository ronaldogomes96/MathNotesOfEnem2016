import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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

#Apagando as colunas de treino que nao existem no teste
treino = treino.drop(colunasInexistentesTest, axis=1)
