'''
Created on 4 de nov de 2017

@author: raul
'''
from abc import ABC,abstractmethod
from bigfloat import BigFloat
import math as m
import mpmath as mp
import numpy as np 


class Funcao(ABC):
    '''
    Classe que representa uma funcao. Vai ser extendida para outras
    '''
        
    @abstractmethod
    def fx(self,x):
        pass
    
    @abstractmethod
    def dFx(self,x):
        pass
    
class Linear(Funcao):
    def fx(self,x):
        return 1 if x >= 0 else 0
    
    def dFx(self,x):
        return 0
    
class Sigmoid(Funcao):
    "Classe que representa a funcao de ativacao sigmoide"
    def __init__(self,a=1):
        self.a = a
    
    def fx(self,x):
        try:
            return 1/(1+m.e**(-1*x*self.a))
        except OverflowError:
            if(x<0):
                return 0
            return 1
            
    def dFx(self,x):
        try:
            return (1-self.fx(x))*self.fx(x)
        except OverflowError:
            if(x<0):
                return 0
            return 1
            
class TanH(Funcao):
    "Classe que representa a funcao de ativacao hiperbolica"
    def fx(self,x):
        return m.tanh(x)
    
    def dFx(self,x):
        return mp.sech(x)**2

class LReLU(Funcao):
    
    def fx(self,x):
        return max([0,0.01*x])
    
    def dFx(self,x):
        if(x<=0):
            return 0.01
        else:
            return 1

class Gaussiana(Funcao):
    
    def fx(self,x,c,r):
        distancia = np.linalg.norm(np.array(c)-np.array(x))
        d = m.pow(distancia,2)
        return m.exp(-(d)/m.pow(r,2))
        

    def dFx(self,x): #procurar derivada depois
        return 1


        