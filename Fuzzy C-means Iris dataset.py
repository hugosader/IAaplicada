# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

import skfuzzy as fuzz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Carrega o iris dataset em iris 
iris = load_iris()

n = len(set(iris.target)) # Quantidade de Clusters (3)

numeroDeTestes=1
for cont in range(numeroDeTestes):


    # Divisão entre treino e teste aleatória (50% para cada)
    X_treino, X_teste, y_treino, y_teste = train_test_split(iris.data, iris.target, test_size=0.5, random_state=4)
    
    # Divisão entre treino e teste usando paridade do indice (50% para cada)
    # X_treino, X_teste, y_treino, y_teste = iris.data[0:][::2], iris.data[1:][::2], iris.target[0:][::2], iris.target[1:][::2]
     
    # Implementa o Algoritmo Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_treino.transpose(), n, 2, error=0.005, maxiter=1000, init=None, seed = 4)
    cluster_membership = np.argmax(u, axis=0)

    # Resultados
    train_labels = y_treino
    u_treino, u0_treino, d_treino, jm_treino, p_treino, fpc_treino = fuzz.cluster.cmeans_predict(
        X_treino.transpose(), cntr, 2, error=0.005, maxiter=1000)
    predictions_train = np.argmax(u_treino, axis=0)
    
    test_labels = y_teste
    u_teste, u0_teste, d_teste, jm_teste, p_teste, fpc_teste = fuzz.cluster.cmeans_predict(
        X_teste.transpose(), cntr, 2, error=0.005, maxiter=1000)
    predictions_test = np.argmax(u_teste, axis=0)
    
    # ================== Clusterização vira Classificação =====================
    # Matriz de Confusão
    mat_treino = confusion_matrix(predictions_train, train_labels)    
    mat_teste = confusion_matrix(predictions_test, test_labels)
    
    # Permutação (dos números que identificam as classes) que maximiza verdadeiros positivos
    PQMVP = np.argmax(mat_treino, axis=1)

    # Renomeando (renumerando) as classes de acordo
    predictions_train = predictions_train + n
    predictions_test  = predictions_test  + n
    for classe in range(n):        
        predictions_train[predictions_train==classe+n] = PQMVP[classe]
        predictions_test [predictions_test ==classe+n] = PQMVP[classe]
    #=========================================================================
    
    # Usando Métricas
    print('\nRESULTADOS DO TREINO\n')
    print('accuracy', accuracy_score(predictions_train, train_labels),'\n')
    print(classification_report(predictions_train, train_labels))
    print('\nRESULTADOS DO TESTE\n')
    print('accuracy', accuracy_score(predictions_test, test_labels),'\n')    
    print(classification_report(predictions_test, test_labels))


