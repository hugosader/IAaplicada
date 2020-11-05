# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:44:35 2020

@author: Ricardo
"""

import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carrega o iris dataset em iris 
iris = load_iris() 

def f(X):
    
    neighb = int(X[0])
     
    # Divisão entre treino e teste aleatória (50% para cada)
    # X_treino, X_teste, y_treino, y_teste = train_test_split(iris.data, iris.target, test_size=0.5)
    
    # Divisão entre treino e teste usando paridade do indice (50% para cada)
    X_treino, X_teste, y_treino, y_teste = iris.data[0:][::2], iris.data[1:][::2], iris.target[0:][::2], iris.target[1:][::2]
     
    # Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors=neighb, weights="uniform")
    neigh.fit(X_treino, y_treino)
    
    
    resEsperado = y_teste
    resObtido = neigh.predict(X_teste)
    
    # Acertos
    acertos=0
    for r in range(len(resEsperado)):
        if resObtido[r]==resEsperado[r]:
            acertos+=1
            
    porcentAcertos=100*acertos/len(resEsperado)
    
    '''
    print("\nResultado Esperado")
    print(resEsperado)
    print("\nResultado Obtido")
    print(resObtido)
    '''
    
    # print("\nPorcentual de acerto: %s" % str(round(porcentAcertos,3)))
    
    # Custo
    J = porcentAcertos
    
    return -J

# Limites do número de vizinhos
varbound=np.array([[1, 70]])

# Parâmetros do algoritmo genético
algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':20,\
                   'mutation_probability':0.5,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

# Modelo
model=ga(function=f,\
            dimension=1,\
            variable_type='int',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

# Execução
model.run()

# Melhor número de vizinhos
res = model.best_variable




