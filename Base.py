'''
Created on Aug 14, 2017

@author: raul
'''
from copy import deepcopy
from random import shuffle
from sklearn import preprocessing

class Base(object):
    
    def __init__(self, classes,atributos,nome=""):
        self.nome = nome
        self.tiposClasses = []
        self.classesOri = deepcopy(classes)
        self.atributos = deepcopy(atributos)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        for e in classes:
            if(e not in self.tiposClasses):
                self.tiposClasses.append(e)
        self.__classeNumericas() 
    
    
    def getSubBaseClasse(self,indice):
        subClasse = []
        subAtributos = []
        for i,e in enumerate(self.classesOri):
            if(e == self.tiposClasses[indice]):
                subClasse.append(e)
                subAtributos.append(self.atributos[i])
        return Base(subClasse,subAtributos)
    
    def __classeNumericas(self):
        self.classes = []
        for i in self.classesOri:
            self.classes.append(self.tiposClasses.index(i))
    
    #embaralha os elementos da base
    def embaralharBase(self):
        c = list(zip(self.classes, self.atributos,self.classesOri))
        shuffle(c)
        self.classes,self.atributos,self.classesOri = zip(*c)
    
    def gerarBaseComMenosAtr(self,qtAtributos):
        novosAtr = []
        for i in self.atributos:
            novosAtr.append(i[0:qtAtributos])
        return Base(self.classesOri,novosAtr)
    
    def normalizar(self):
        self.atributosOri = deepcopy(self.atributos)
        self.atributos = self.min_max_scaler.fit_transform(self.atributos)
    
    def desnormalizar(self):
        self.atributos = self.atributosOri
        
        
    def copy(self):
        return Base(self.classes,self.atributos)
    

        
        
            
        
        

                
        
        