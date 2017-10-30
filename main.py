'''
Created on 25 de out de 2017

@author: raul
'''
from MLP import MLP
from Base import Base
from CreateBaseFromFile import CreateBaseFromFile as cbff
from sklearn.neural_network import MLPClassifier
from random import shuffle
if __name__ == '__main__':
    
    bWine = cbff.createFromFile("Base/wine",[0],[])
    
    bWine1 = bWine.getSubBaseClasse(0)
    bWine2 = bWine.getSubBaseClasse(1)
    bWine3 = bWine.getSubBaseClasse(2)
    
    qtBase = len(bWine1.classes)
    qtBase2 = len(bWine2.classes)
    qtBase3 = len(bWine3.classes)
    
    tc = bWine2.classes[0:int(qtBase2/2)]+bWine3.classes[0:int(qtBase3/2)]+bWine1.classes[0:int(qtBase/2)]
    ta = bWine2.atributos[0:int(qtBase2/2)]+bWine3.atributos[0:int(qtBase3/2)]+bWine1.atributos[0:int(qtBase/2)]
    
    c = list(zip(tc, ta))
    shuffle(c)
    tc,ta = zip(*c)
    bWineTreino = Base(tc,ta)
  

    tc = bWine1.classes[int(qtBase/2):]+bWine2.classes[int(qtBase2/2):]+bWine3.classes[int(qtBase3/2):]
    ta = bWine1.atributos[int(qtBase/2):]+bWine2.atributos[int(qtBase2/2):]+bWine3.atributos[int(qtBase3/2):]
    bWineTeste = Base(tc,ta)

    mlpsk = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 3), random_state=1)
    #mlpsk.fit(bWineTreino.atributos,bWineTreino.classes)
    w = [len(bWine.atributos[0])+1,4]
    mlp = MLP([3,3],w,2,100000,tipoFunc="sigmoid",a=1)
    mlp.fit(bWineTreino)
    erro = 0
   
    for i,atr in enumerate(bWineTeste.atributos):
        print(atr)
        #r = mlpsk.predict(atr)
        r = mlp.avaliar(atr)
        c = mlp.decimalParaBin(int(bWineTeste.classes[i]))
        print(r,c)
        if(r != c):
            print("EEEEEEEEEEEEROUUUUUUUUUUUU")
            erro+=1
    print(erro/len(bWineTeste.classes))

    pass