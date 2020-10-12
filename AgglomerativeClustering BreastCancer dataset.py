# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:20:03 2020

@author: Ricardo
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


#Carrega o iris dataset em iris 
iris = load_breast_cancer() 

n_testes_uteis=0
numeroDeTestes=100
desempenho=0 
for cont in range(numeroDeTestes):
    

    #Divisão entre treino e teste aleatória (50% para cada)
    X_treino, X_teste, y_treino, y_teste = train_test_split(iris.data, iris.target, test_size=0.5)
    
    #Divisão entre treino e teste usando paridade do indice (50% para cada)
    #X_treino, X_teste, y_treino, y_teste = iris.data[0:][::2], iris.data[1:][::2], iris.target[0:][::2], iris.target[1:][::2]
     
    #Implementa o Algoritmo AffinityPropagation
    af = AgglomerativeClustering(n_clusters=2, linkage='ward', affinity='euclidean').fit(X_treino)
    
    #affinity{“euclidean”, “l1”, “l2”, “manhattan”, “cosine”}
    #linkage{“ward”, “complete”, “average”, “single”}
    
    resEsperado = y_teste
    resObtido = af.fit_predict(X_teste)
    
    
    acertos=0
    for r in range(len(resEsperado)):
        if resObtido[r]==resEsperado[r]:
            acertos+=1
            
    porcentAcertos=acertos/len(resEsperado)
    
    #if porcentAcertos < 0.5:
    #    porcentAcertos=1-porcentAcertos
    
    #'''
    print("\nResultado Esperado")
    print(resEsperado)
    print("\nResultado Obtido")
    print(resObtido)
    #'''
    
    print("\nPorcentual de acerto: %s" % str(round(100*porcentAcertos,3)))
    
    if porcentAcertos>0.5 and n_testes_uteis<10:
        desempenho+=100*porcentAcertos
        n_testes_uteis+=1
    
desempenho=desempenho/10
print("\nDesempenho: %s" % str(round(desempenho,3)))