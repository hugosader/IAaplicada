# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

#Carrega o iris dataset em iris 
iris = load_breast_cancer()

n_testes_uteis=0
numeroDeTestes=100
desempenho=0 
for cont in range(numeroDeTestes):

    ncenters = 2

    #Divisão entre treino e teste aleatória (50% para cada)
    X_treino, X_teste, y_treino, y_teste = train_test_split(iris.data, iris.target, test_size=0.5)
    
    #Divisão entre treino e teste usando paridade do indice (50% para cada)
    #X_treino, X_teste, y_treino, y_teste = iris.data[0:][::2], iris.data[1:][::2], iris.target[0:][::2], iris.target[1:][::2]
     
    #Implementa o Algoritmo Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_treino.transpose(), ncenters, 2, error=0.005, maxiter=1000, seed=0)
    cluster_membership = np.argmax(u, axis=0)
    
    
    resEsperado = y_teste
    u_teste, u0_teste, d_teste, jm_teste, p_teste, fpc_teste = fuzz.cluster.cmeans_predict(
        X_teste.transpose(), cntr, 2, error=0.005, maxiter=1000)
    resObtido = np.argmax(u_teste, axis=0)
    
    
    acertos=0
    for r in range(len(resEsperado)):
        if resObtido[r]==resEsperado[r]:
            acertos+=1
            
    porcentAcertos=acertos/len(resEsperado)
    
    #'''
    print("Resultado Esperado")
    print(resEsperado)
    print("Resultado Obtido")
    print(resObtido)
    #'''
    
    print("\nPorcentual de acerto: %s" % str(round(100*porcentAcertos,3)))
    
    if porcentAcertos>0.5 and n_testes_uteis<10:
        desempenho+=100*porcentAcertos
        n_testes_uteis+=1
    
desempenho=desempenho/10
print("\nDesempenho: %s" % str(round(desempenho,3)))

