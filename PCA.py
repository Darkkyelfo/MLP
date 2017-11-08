'''
Created on Aug 29, 2017

@author: raul
'''
import numpy as np
from Base import Base
import math
import copy
from numba import jit

class PCA(object):
    
    def run (self,base1,k="T"):
        if(k=="T"):
            k = len(base1.atributos[0])

        copia = np.array(copy.deepcopy(base1.atributos))
        autoVectors = self.autoVectors[0:k]
        novosAtributos = np.dot(copia,np.array(autoVectors).T)
        
        return (Base(base1.classes,novosAtributos))
    
    def fit(self,bTreino):
        copia = np.array(copy.copy(bTreino.atributos))
        cov = np.cov(copia.T)
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = autoVectors.T
        self.autoValues,self.autoVectors = zip(*sorted(zip(autoValues, autoVectors),reverse=True))
    
        
class PCA_SCORE(PCA):
    
    def fit(self,base1):
        super().fit(base1)
        base = super().run(base1, len(base1.classes))
        m1 = base.getSubBaseClasse(0)
        m2 = base.getSubBaseClasse(1)
        m1 = np.mean(m1.atributos, axis=0)
        m2 = np.mean(m2.atributos, axis=0)
        scores = self.__score(m1, m2)
        self.scores,self.autoVectors = zip(*sorted(zip(scores, self.autoVectors ),reverse=True))
    
    @jit
    def __score(self,media1,media2):
        scores = []
        for i in range(len(media1)):
            s = 0
            if(self.autoValues[i] != 0):
                s = math.fabs((media1[i]-media2[i]))/self.autoValues[i]
            scores.append(s)
        return scores

  
               
    
    