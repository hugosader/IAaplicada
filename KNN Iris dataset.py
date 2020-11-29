# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
     
    # Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors=20, weights="uniform")
    neigh.fit(X_treino, y_treino)
    
    # Resultados
    train_labels = y_treino
    predictions_train = neigh.predict(X_treino)
    
    test_labels = y_teste
    predictions_test = neigh.predict(X_teste)    
    
    # Usando Métricas
    print('\nRESULTADOS DO TREINO\n')
    print('accuracy', accuracy_score(predictions_train, train_labels),'\n')
    print(classification_report(predictions_train, train_labels))
    print('\nRESULTADOS DO TESTE\n')
    print('accuracy', accuracy_score(predictions_test, test_labels),'\n')    
    print(classification_report(predictions_test, test_labels))
        
    
    