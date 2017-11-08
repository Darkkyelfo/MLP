'''
Created on 4 de nov de 2017

@author: raul
'''
from abc import ABC,abstractmethod
from bigfloat import BigFloat
import math as m
import mpmath as mp


class Funcao(ABC):
    '''
    Classe que representa uma funcao. Vai ser extendida para outras
    '''
        
    @abstractmethod
    def fx(self):
        pass
    
    @abstractmethod
    def dFx(self):
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
        return 1/(1+m.e**(-1*x*self.a))

    def dFx(self,x):
        return (1-self.fx(x))*self.fx(x)

    
class TanH(Funcao):
    "Classe que representa a funcao de ativacao hiperbolica"
    def fx(self,x):
        return m.tanh(x)
    
    def dFx(self,x):
        return mp.sech(x)**2

class LReLU(Funcao):
    
    def fx(self,x):
        return 1 if x >= 0 else 0.01
    
    def dFx(self,x):
        return self.fx(x)
    
        