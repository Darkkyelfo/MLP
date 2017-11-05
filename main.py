'''
Created on 25 de out de 2017

@author: raul
'''
from neuralNetwork import MLP
from Base import Base
from CreateBaseFromFile import CreateBaseFromFile as cbff
from PCA import PCA as PCA
import numpy as np
if __name__ == '__main__':
    
    bWine = cbff.createFromFile("Base/wine",[0],[])
    bIris = cbff.createFromFile("Base/iris.csv", [4], [])

    
    bWine1 = bWine.getSubBaseClasse(0)
    bWine2 = bWine.getSubBaseClasse(1)
    bWine3 = bWine.getSubBaseClasse(2)
    
    bIris1 = bIris.getSubBaseClasse(0)
    bIris2 = bIris.getSubBaseClasse(1)
    bIris3 = bIris.getSubBaseClasse(2)
    
    qtBase = len(bWine1.classesOri)
    qtBase2 = len(bWine2.classesOri)
    qtBase3 = len(bWine3.classesOri)
    
    qtIris1 = len(bIris1.classes)
    qtIris2 = len(bIris2.classes)
    qtIris3 = len(bIris3.classes)
    
    tc = bWine1.classesOri[0:int(qtBase/2)]+bWine2.classesOri[0:int(qtBase2/2)]+bWine3.classesOri[0:int(qtBase2/2)]
    ta = bWine1.atributos[0:int(qtBase/2)]+bWine2.atributos[0:int(qtBase2/2)]+bWine3.atributos[0:int(qtBase2/2)]
    
    pca = PCA()
    bWineTreino = Base(tc,ta)
    pca.fit(bWineTreino)
    bWineTreino = pca.run(bWineTreino, 13)
    bWineTreino.embaralharBase()
    
    tc = bWine1.classesOri[int(qtBase/2):]+bWine2.classesOri[int(qtBase2/2):]+bWine3.classesOri[int(qtBase3/2):]
    ta = bWine1.atributos[int(qtBase/2):]+bWine2.atributos[int(qtBase2/2):]+bWine3.atributos[int(qtBase3/2):]
    
    bWineTeste = pca.run(Base(tc,np.array(ta)),13)
    
    #base iris treino
    
    tc = bIris1.classesOri[0:int(qtIris1/2)]+bIris2.classesOri[0:int(qtIris2/2)]+bIris3.classesOri[0:int(qtIris3/2)]
    ta = bIris1.atributos[0:int(qtIris1/2)]+bIris2.atributos[0:int(qtIris2/2)]+bIris3.atributos[0:int(qtIris3/2)]
    
    bIrisTreino = Base(tc,ta)
    #print(bIrisTreino.classes)
    bIrisTreino.embaralharBase()
    #embaralha o conjuto de treino
    #c = list(zip(bIrisTreino.classes, bIrisTreino.atributos,bIrisTreino.classesOri))
   # shuffle(c)
    #bIrisTreino.classes,bIrisTreino.atributos,bIrisTreino.classesOri = zip(*c)
    
    #bIrisTreino = Base(tc,ta)
    
    tc = bIris1.classesOri[int(qtIris1/2):]+bIris2.classesOri[int(qtIris2/2):]+bIris3.classesOri[int(qtIris3/2):]
    ta = bIris1.atributos[int(qtIris1/2):]+bIris2.atributos[int(qtIris2/2):]+bIris3.atributos[int(qtIris3/2):]
    
    bIrisTeste = Base(tc,ta)
    #print(bIrisTeste.classes)


     
    #mlpsk = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 3), random_state=1)
    #mlpsk.fit(bWineTreino.atributos,bWineTreino.classesOri)
    w = [len(bWine.atributos[0])+1,5]
    w1 = [len(bIris.atributos[0])+1,5]
    """
    #Rede neural pra base wine
    
    mlp = MLP([4,3],w,0.7,10000,tipoFunc="sigmoid",a=1)
    mlp.fit(bWineTreino)
    erro = 0
    
    for i,atr in enumerate(bWineTeste.atributos):
        print(atr)

        r = mlp.avaliar(atr)
        c = mlp.decimalParaBin(int(bWineTeste.classes[i]))
        print(r,c)
        if(r != c):
            print("EEEEEEEEEEEEROUUUUUUUUUUUU")
            erro+=1
    print("erro %s"%(erro/len(bWineTeste.classesOri)))
    input("precione para seguir")
    """
    
    #Rede neural pra base Iris
    mlpIris = MLP([4,3],w1,0.2,10000,tipoFunc="Sigmoid")
    mlpIris.fit(bIrisTreino)
    #mlpIris.carregarDeArquivo("redes/iris1")
    erro = 0
    for i,atr in enumerate(bIrisTeste.atributos):
        r = mlpIris.avaliar(atr)
        c = mlpIris.decimalParaBin(int(bIrisTeste.classes[i]))
        print(atr)
        print(r,c)
        if(r != c):
            print("EEEEEEEEEEEEROUUUUUUUUUUUU")
            erro+=1
        print("\n")
    print("erro iris %s%%"%((erro/len(bIrisTeste.classesOri))*100))
    #mlpIris.salveEmArquivo("redes/iris2")

    pass