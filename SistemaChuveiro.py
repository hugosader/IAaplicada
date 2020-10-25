# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:13:27 2020

@author: Ricardo
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Entradas
tempAgua = ctrl.Antecedent(np.arange(0, 51, 1), 'tempAgua')
tempAmbiente = ctrl.Antecedent(np.arange(0, 51, 1), 'tempAmbiente')
                               
# Saída
vazaoTorneira = ctrl.Consequent(np.arange(0, 101, 1), 'vazaoTorneira')

# Funções de pertinência
tempAgua['baixa'] = fuzz.trimf(tempAgua.universe, [0, 0, 25])
tempAgua['media'] = fuzz.trimf(tempAgua.universe, [0, 25, 50])
tempAgua['alta'] = fuzz.trimf(tempAgua.universe, [25, 50, 50])

tempAmbiente['baixa'] = fuzz.trimf(tempAmbiente.universe, [0, 0, 25])
tempAmbiente['media'] = fuzz.trimf(tempAmbiente.universe, [0, 25, 50])
tempAmbiente['alta'] = fuzz.trimf(tempAmbiente.universe, [25, 50, 50])

vazaoTorneira['muitoBaixa'] = fuzz.trimf(vazaoTorneira.universe, [0, 0, 25])
vazaoTorneira['baixa'] = fuzz.trimf(vazaoTorneira.universe, [0, 25, 50])
vazaoTorneira['media'] = fuzz.trimf(vazaoTorneira.universe, [25, 50, 75])
vazaoTorneira['alta'] = fuzz.trimf(vazaoTorneira.universe, [50, 75, 100])
vazaoTorneira['muitoAlta'] = fuzz.trimf(vazaoTorneira.universe, [75, 100, 100])

# Regras de inferência 
rule1 = ctrl.Rule(tempAgua['baixa'] & tempAmbiente['baixa'], vazaoTorneira['muitoBaixa'])

rule2 = ctrl.Rule(tempAgua['media'] & tempAmbiente['media'], vazaoTorneira['media'])

rule3 = ctrl.Rule(tempAgua['alta'] & tempAmbiente['alta'], vazaoTorneira['muitoAlta'])

rule4 = ctrl.Rule(tempAgua['media'] & tempAmbiente['alta'], vazaoTorneira['alta'])
rule5 = ctrl.Rule(tempAgua['alta'] & tempAmbiente['media'], vazaoTorneira['alta'])

rule6 = ctrl.Rule(tempAgua['media'] & tempAmbiente['baixa'], vazaoTorneira['baixa'])
rule7 = ctrl.Rule(tempAgua['baixa'] & tempAmbiente['media'], vazaoTorneira['baixa'])

rule8 = ctrl.Rule(tempAgua['baixa'] & tempAmbiente['alta'], vazaoTorneira['media'])
rule9 = ctrl.Rule(tempAgua['alta'] & tempAmbiente['baixa'], vazaoTorneira['media'])

# Sistema de controle
chuveiro_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Simulação
chuveiro = ctrl.ControlSystemSimulation(chuveiro_ctrl)

# Aplicando entradas
chuveiro.input['tempAgua'] = 35
chuveiro.input['tempAmbiente'] = 25
chuveiro.compute()

#Gráficos
tempAgua.view()
tempAmbiente.view()
vazaoTorneira.view()

#Defuzzyficação
vazaoTorneira.view(sim=chuveiro)

print('\nA vazao da torneira será de '+str(round(chuveiro.output['vazaoTorneira'], 3))+" %\n")
