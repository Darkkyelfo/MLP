'''
Created on 25 de out de 2017

@author: raul
'''
from neuralNetwork import MLP,MLPBatch,RBF
from Base import Base
from CreateBaseFromFile import CreateBaseFromFile as cbff
if __name__ == '__main__':
    
    bIris = cbff.createFromFile("Base/iris.csv", [4], [])

  
    bIris1 = bIris.getSubBaseClasse(0)
    bIris2 = bIris.getSubBaseClasse(1)
    bIris3 = bIris.getSubBaseClasse(2)
    
    
    qtIris1 = len(bIris1.classes)
    qtIris2 = len(bIris2.classes)
    qtIris3 = len(bIris3.classes)
    
  
    #base iris treino
    
    tc = bIris1.classesOri[0:int(qtIris1/2)]+bIris2.classesOri[0:int(qtIris2/2)]+bIris3.classesOri[0:int(qtIris3/2)]
    ta = bIris1.atributos[0:int(qtIris1/2)]+bIris2.atributos[0:int(qtIris2/2)]+bIris3.atributos[0:int(qtIris3/2)]
    
    bIrisTreino = Base(tc,ta)
    bIrisTreinoNaoE = bIrisTreino.copy()

    bIrisTreino.embaralharBase()
    
    tc = bIris1.classesOri[int(qtIris1/2):]+bIris2.classesOri[int(qtIris2/2):]+bIris3.classesOri[int(qtIris3/2):]
    ta = bIris1.atributos[int(qtIris1/2):]+bIris2.atributos[int(qtIris2/2):]+bIris3.atributos[int(qtIris3/2):]
    
    bIrisTeste = Base(tc,ta)
         
    #Rede neural pra base wine
    '''
    rbf = RBF(2)
    bXor = Base([0,1,1,0],[[0,0],[0,1],[1,0],[1,1]])
    
    rbf.fitDDA(bXor)
    '''
    rbfIris = RBF(3)
    rbfIris.fitDDA(bIrisTreinoNaoE) 
    for cam in rbfIris.camIntermediaria:
        print(len(cam))
    print(rbfIris.testar(bIrisTeste))
    
    pass