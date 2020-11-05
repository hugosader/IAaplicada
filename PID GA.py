# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:44:35 2020

@author: Ricardo
"""

import control as co
import numpy as np
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga


# Diferencial de tempo
dt = 0.001

# Vetor de tempo 
t0=0
tf=20   
t = np.linspace(t0, tf, int((tf-t0)/dt+1))

# Definindo a variável de frequência s
s = co.tf('s')

# Função de transferência de malha aberta do processo
H = 2/(4*s+1)

def f(X):
        
    Kp, Ki, Kd = X[0], X[1], X[2]
    
    # Controlador
    K = Kp + Ki/s + Kd*s/(1+0.001*s)
    
    # Função de transferência de malha fechada r(t) -> [FTMF] -> y(t)
    FTMF = co.feedback(co.series(K, H))
        
    # Resposta ao degrau
    t1, y1 = co.step_response(FTMF, T=t,  X0=0.0)
    # plt.plot(t1, y1, label='Resposta ao degrau')

    # Erro
    erro = 1 - y1
    # plt.plot(t1, erro**2, label='Erro quadrático')
    # plt.plot(t1, abs(erro), label='Erro absoluto')
    
    # Plot
    # plt.xlabel('Tempo(s)')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.legend()
    # plt.show()
    
    # u(t)
    t2, u1, xout = co.forced_response(K, T=t1, U=erro)
    # plt.plot(t2, u1, label='Sinal de controle')
    
    # Plot
    # plt.xlabel('Tempo(s)')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.legend()
    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((x1,5,y1,y2))
    # plt.show()
    
    # Integral de erro quadrático
    I = sum(erro**2)*dt
    
    # Integral de erro absoluto
    # I = sum(abs(erro))*dt
    
    # Custo (Função LQR)
    Q = 1
    R = 0.01
    
    J = Q*I + R*sum(u1**2)*dt
    # print('\n', J)    
    
    return J

# Limites de Kp, Ki, Kd
varbound=np.array([[0, 10]]*3)

# Parâmetros do algoritmo genético
algorithm_param = {'max_num_iteration': 50,\
                   'population_size':20,\
                   'mutation_probability':0.25,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

# Modelo
model=ga(function=f,\
            dimension=3,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

# Execução
model.run()

# Melhor Kp, Ki, Kd
res = model.best_variable
Kp, Ki, Kd = res[0], res[1], res[2]

# Controlador ideal
K_ = Kp + Ki/s + Kd*s

# Controlador
K = Kp + Ki/s + Kd*s/(1+0.0001*s)

# Função de transferência de malha fechada r(t) -> [FTMF] -> y(t)
FTMF = co.feedback(co.series(K,H))

# print('\nH = \n', H)
# print('K_ = \n', K_)
# print('K = \n', K)
# print('FTMF = \n', FTMF)

# Resposta ao degrau
t1, y1 = co.step_response(FTMF, T=t,  X0=0.0)
plt.plot(t1, y1, label='Resposta ao degrau')

# Erro
erro = 1 - y1
plt.plot(t1, erro**2, label='Erro quadrático')
plt.plot(t1, abs(erro), label='Erro absoluto')

# Plot
plt.xlabel('Tempo(s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.show()

# u(t)
t2, u1, xout = co.forced_response(K, T=t, U=erro)
plt.plot(t2, u1, label='Sinal de controle')

# Plot
plt.xlabel('Tempo(s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,int(x2/4),y1,y2))
plt.show()


