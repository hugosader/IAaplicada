# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:13:27 2020

@author: Ricardo
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.datasets import load_iris

iris=load_iris()

#Divisão entre treino e teste usando paridade do indice (50% para cada)
X_treino, X_teste, y_treino, y_teste = iris.data[0:][::2], iris.data[1:][::2], iris.target[0:][::2], iris.target[1:][::2]


'''
#min max sepalLength
print(min(iris.data[:,0])) #4.3 [4.0 8.0]
print(max(iris.data[:,0])) #7.9
#min max sepalWidth
print(min(iris.data[:,1])) #2.0 [2.0 5.0]
print(max(iris.data[:,1])) #4.4
#min max petalLength
print(min(iris.data[:,2])) #1.0 [1.0 7.0]
print(max(iris.data[:,2])) #6.9
#min max petalWidth
print(min(iris.data[:,3])) #0.1 [0.0 3.0]
print(max(iris.data[:,3])) #2.5
'''

# Entradas
sepalLength = ctrl.Antecedent(np.arange(4, 8.1, 0.1), 'sepalLength')
sepalWidth = ctrl.Antecedent(np.arange(2, 5.1, 0.1), 'sepalWidth')
petalLength = ctrl.Antecedent(np.arange(1, 7.1, 0.1), 'petalLength')
petalWidth = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'petalWidth')

# Saída
species = ctrl.Consequent(np.arange(0, 31, 1), 'species')

# Funções de pertinência
sepalLength['small'] = fuzz.trimf(sepalLength.universe, [4, 4, 6.55])
sepalLength['average'] = fuzz.trimf(sepalLength.universe, [6, 6.55, 7.1])
sepalLength['big'] = fuzz.trimf(sepalLength.universe, [6.55, 8, 8])

sepalWidth['small'] = fuzz.trimf(sepalWidth.universe, [2, 2, 2.75])
sepalWidth['average'] = fuzz.trimf(sepalWidth.universe, [2.4, 2.75, 3.1])
sepalWidth['big'] = fuzz.trimf(sepalWidth.universe, [2.75, 5, 5])

petalLength['small'] = fuzz.trimf(petalLength.universe, [1, 1, 3.75])
petalLength['average'] = fuzz.trimf(petalLength.universe, [2.5, 3.75, 5])
petalLength['big'] = fuzz.trimf(petalLength.universe, [3.75, 7, 7])

petalWidth['small'] = fuzz.trimf(petalWidth.universe, [0, 0, 1.5])
petalWidth['average'] = fuzz.trimf(petalWidth.universe, [1, 1.5, 1.8])
petalWidth['big'] = fuzz.trimf(petalWidth.universe, [1.5, 3, 3])


species['Setosa'] = fuzz.trimf(species.universe, [0, 0, 15])
species['Versicolor'] = fuzz.trimf(species.universe, [10, 15, 20])
species['Virginica'] = fuzz.trimf(species.universe, [15, 30, 30])

# Regras de inferência (obtidas através de análises/comparações feitas entre 
# iris.data e iris.target da parte reservada a treino, X_treino e y_treino)
rule1 = ctrl.Rule(sepalLength["big"], species['Virginica'])

rule2 = ctrl.Rule(sepalWidth["small"], species['Versicolor'])
rule3 = ctrl.Rule(sepalWidth["small"] | sepalWidth["average"] , species['Versicolor'])

rule4 = ctrl.Rule(petalLength["small"], species['Setosa'])
rule5 = ctrl.Rule(petalLength["average"], species['Versicolor'])
rule6 = ctrl.Rule(petalLength["big"], species['Virginica'])

rule7 = ctrl.Rule(petalWidth["small"], species['Setosa'])
rule8 = ctrl.Rule(petalWidth["average"], species['Versicolor'])
rule9 = ctrl.Rule(petalWidth["big"], species['Virginica'])

# Sistema de controle
iris_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Simulação
simulation = ctrl.ControlSystemSimulation(iris_ctrl)

#TREINO
print('\nTREINO')

# Aplicando entradas
tests = X_treino
resultado = []
for test in tests:
    simulation.input['sepalLength'] = test[0]
    simulation.input['sepalWidth'] = test[1]
    simulation.input['petalLength'] = test[2]
    simulation.input['petalWidth'] = test[3]
    simulation.compute()
    
    #Defuzzyficação TREINO  
    especies = 1
    if round(simulation.output['species'])<10:
        especies = 0
    elif round(simulation.output['species'])>20:
        especies = 2
    resultado.append(especies)

resultado = np.asarray(resultado)
print('\nResultado Esperado:')
print(y_treino)
print('\nResultado Obtido:')
print(resultado)
 
# Calculo do porcentual de acerto TREINO              
acertos=0
for r in range(len(y_treino)):
    if resultado[r]==y_treino[r]:
        acertos+=1
            
porcentAcertos=acertos/len(y_treino)
print('\n porcentual de acerto: '+str(100*porcentAcertos)+' %\n')

# TESTES
print('\nTESTES')

# Aplicando entradas
tests = X_teste
resultado = []
for test in tests:
    simulation.input['sepalLength'] = test[0]
    simulation.input['sepalWidth'] = test[1]
    simulation.input['petalLength'] = test[2]
    simulation.input['petalWidth'] = test[3]
    simulation.compute()

    #Defuzzyficação TESTES    
    especies = 1
    if round(simulation.output['species'])<10:
        especies = 0
    elif round(simulation.output['species'])>20:
        especies = 2
    resultado.append(especies)

resultado = np.asarray(resultado)
print('\nResultado Esperado:')
print(y_teste)
print('\nResultado Obtido:')
print(resultado)

# Calculo do porcentual de acerto TESTES             
acertos=0
for r in range(len(y_teste)):
    if resultado[r]==y_teste[r]:
        acertos+=1
           
porcentAcertos=acertos/len(y_teste)
print('\n Porcentual de acerto: '+str(100*porcentAcertos)+' %\n')

# Gráficos
sepalLength.view()
sepalWidth.view()
petalLength.view()
petalWidth.view()
species.view()
# species.view(sim=simulation)



