'''
Created on 7 de out de 2017

@author: raul
'''
from numba import jit
from copy import deepcopy
from math import e as euler
class Perceptron(object):
    '''
    classdocs
    '''
    
    def __init__(self,weights,n = 1,qtIteracoes = 10,e = 0,bias = -1,stepFunction = "1 if x >= 0 else 0"):
        self.__stepFunction = stepFunction
        self.weights = deepcopy(weights) 
        self.n = n
        self.bias = bias
        self.qtIteracoes = qtIteracoes
        self.e = e
        self.erros = []
    
    #algumas funcoes tem um valor fixo que as modifica com a sigmoid
    #1/(1+e**(-1*x*a)) geralmente usam esse a como 1
    def setStep(self,funcao,a=1):
        self.__stepFunction = funcao
        self.a = a
    
    def step(self,x):
        try:
            x = x
            a = self.a
            e = euler #numero de euler
            return eval(self.__stepFunction)
        except:
            if(x<0):
                return 0
            else:
                return 1

    def __somatorio(self,args):
        resultado = 0
        for i,x in enumerate(args):
            resultado = self.weights[i]*x + resultado
        resultado = self.weights[i+1]*self.bias + resultado
        self.resultoSomatorio = resultado
        return resultado
    
    
    def trainFunction(self,atributos,cAchada,cReal):
        for i,atr in enumerate(atributos):
            self.weights[i] = self.weights[i]+self.n*(cReal-cAchada)*atr
        self.weights[len(atributos)-1] = self.weights[len(atributos)-1]+self.n*(cReal-cAchada)*self.bias
   
    @jit
    def treinar(self,cTreino):
        tBase = len(cTreino.atributos)
        self.erros = []
        for i in range(self.qtIteracoes):
            erro = 0
            #continuar = False
            for i,atr in enumerate(cTreino.atributos):
                r = self.avaliar(atr)
                if(r != cTreino.classes[i]):
                    self.trainFunction(atr, r, cTreino.classes[i])
                    erro = erro + 1
                    #continuar  = True
            self.erros.append(erro/tBase)
          #  if(erro/tBase<=self.e or continuar==False):
               # break
            
    def avaliar(self,atributos):
        #print(atributos)
        self.entradas = [] #guarda a entrada dos dados
        for i in atributos:
            self.entradas.append(i)
        #print(atributos,"somatorio",self.__somatorio(atributos),"pesos",self.weights)
        #print("step",self.step(self.__somatorio(atributos)))
        self.saida = self.__discretizarSaida(self.step(self.__somatorio(atributos))) #guarda a saida encontrada
        
        return self.saida
                
    #arredonda caso se esteja usando alguma função de ativação continua 
    def __discretizarSaida(self,saida):
        if(saida >= 0.96):
            return 1
        elif(saida <= 0.1):
            return 0
                
        return saida  
        
        
        
            
            
        
        