# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:23:04 2020
https://www.python-course.eu/neural_networks_with_scikit.php
https://medium.com/as-m%C3%A1quinas-que-pensam/m%C3%A9tricas-comuns-em-machine-learning-como-analisar-a-qualidade-de-chat-bots-inteligentes-m%C3%A9tricas-1ba580d7cc96
https://medium.com/@vitorborbarodrigues/m%C3%A9tricas-de-avalia%C3%A7%C3%A3o-acur%C3%A1cia-precis%C3%A3o-recall-quais-as-diferen%C3%A7as-c8f05e0a513c
@author: 
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Carrega o iris dataset em iris 
iris = load_iris()

numeroDeTestes=1
for cont in range(numeroDeTestes):

    # Divisão entre treino e teste aleatória (50% para cada)
    X_treino, X_teste, y_treino, y_teste = train_test_split(iris.data, iris.target, test_size=0.5, random_state=4)
    
    # Divisão entre treino e teste usando paridade do indice (50% para cada)
    # X_treino, X_teste, y_treino, y_teste = iris.data[0:][::2], iris.data[1:][::2], iris.target[0:][::2], iris.target[1:][::2]
    
    # Dimensionando os Dados
    scaler = StandardScaler()    
    scaler.fit(X_treino)    
    X_treino = scaler.transform(X_treino)
    X_teste = scaler.transform(X_teste)       
    
    # Criando o Modelo
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=4)
    
    # Treinando o Modelo
    mlp.fit(X_treino, y_treino)
    
    # Resultados
    train_labels = y_treino
    predictions_train = mlp.predict(X_treino)
    test_labels = y_teste
    predictions_test = mlp.predict(X_teste)
    
    # Usando Métricas
    print('\nRESULTADOS DO TREINO\n')
    print('accuracy', accuracy_score(predictions_train, train_labels),'\n')
    print(classification_report(predictions_train, train_labels))
    print('\nRESULTADOS DO TESTE\n')
    print('accuracy', accuracy_score(predictions_test, test_labels),'\n')    
    print(classification_report(predictions_test, test_labels))
