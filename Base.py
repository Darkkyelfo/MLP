'''
Created on Aug 14, 2017

@author: raul
'''
from copy import deepcopy
class Base(object):
    
    def __init__(self, classes,atributos):
        self.tiposClasses = []
        self.classes = deepcopy(classes)
        self.atributos = deepcopy(atributos)
        for e in classes:
            if(e not in self.tiposClasses):
                self.tiposClasses.append(e) 
    
    
    def getSubBaseClasse(self,indice):
        subClasse = []
        subAtributos = []
        for i,e in enumerate(self.classes):
            if(e == self.tiposClasses[indice]):
                subClasse.append(e)
                subAtributos.append(self.atributos[i])
        return Base(subClasse,subAtributos)
    

        
        
            
        
        

                
        
        