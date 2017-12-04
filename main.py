'''
Created on 25 de out de 2017

@author: raul
'''
from neuralNetwork import MLP,MLPBatch
from Base import Base
from CreateBaseFromFile import CreateBaseFromFile as cbff
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
    

    bWineTreino = Base(tc,ta)
    bWineTreinoNaoEmb = bWineTreino.copy()

    bWineTreino.embaralharBase()
    
    tc = bWine1.classesOri[int(qtBase/2):]+bWine2.classesOri[int(qtBase2/2):]+bWine3.classesOri[int(qtBase3/2):]
    ta = bWine1.atributos[int(qtBase/2):]+bWine2.atributos[int(qtBase2/2):]+bWine3.atributos[int(qtBase3/2):]
    
    bWineTeste = Base(tc,np.array(ta))
    
    #base iris treino
    
    tc = bIris1.classesOri[0:int(qtIris1/2)]+bIris2.classesOri[0:int(qtIris2/2)]+bIris3.classesOri[0:int(qtIris3/2)]
    ta = bIris1.atributos[0:int(qtIris1/2)]+bIris2.atributos[0:int(qtIris2/2)]+bIris3.atributos[0:int(qtIris3/2)]
    
    bIrisTreino = Base(tc,ta)
    bIrisTreinoNaoE = bIrisTreino.copy()

    bIrisTreino.embaralharBase()
    
    tc = bIris1.classesOri[int(qtIris1/2):]+bIris2.classesOri[int(qtIris2/2):]+bIris3.classesOri[int(qtIris3/2):]
    ta = bIris1.atributos[int(qtIris1/2):]+bIris2.atributos[int(qtIris2/2):]+bIris3.atributos[int(qtIris3/2):]
    
    bIrisTeste = Base(tc,ta)
         
    w = [len(bWine.atributos[0])+1,5]
    w1 = [len(bIris.atributos[0])+1,5]
    
    #Rede neural pra base wine
    epocas = 10000
    taxa = 0.001
    
    #Iris
    mlpIris = MLP([4,3],w1,taxa,epocas,tipoFunc="Sigmoid")
    mlpIris.salveEmArquivo("redes/iris1")
    mlpIrisBatch = MLPBatch([4,3],w1,taxa,epocas,tipoFunc="Sigmoid")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    #mlpIris.fit(bIrisTreino)
    #print("sigmoid - estocastica - embaralhada - Não normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreino)
    print("sigmoid - batch - embaralhada - Não normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.fit(bIrisTreinoNaoE)
    print("sigmoid - estocastica - nao embaralhada - Não normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreinoNaoE)
    print("sigmoid - batch - nao embaralhada - nao normalizado:%s" %mlpIrisBatch.test(bIrisTeste))
    
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.setFuncaoAtivacao("lRELU")
    mlpIrisBatch.setFuncaoAtivacao("lRELU")
    
    mlpIris.fit(bIrisTreino)
    print("lRelu - estocastica - embaralhada - Não normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreino)
    print("lRelu - batch - embaralhada - Não normalizado:%s"%mlpIrisBatch.test(bIrisTeste))
    
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.setFuncaoAtivacao("lRELU")
    mlpIrisBatch.setFuncaoAtivacao("lRELU")
    
    mlpIris.fit(bIrisTreinoNaoE)
    print("lRelu - estocastica - nao embaralhada - Não normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreinoNaoE)
    print("lRelu - batch - nao embaralhada - Não normalizado:%s"%mlpIrisBatch.test(bIrisTeste))

    bIrisTreino.normalizar()
    bIrisTeste.normalizar()
    bIrisTreinoNaoE.normalizar()
    
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.fit(bIrisTreino)
    print("sigmoid - estocastica - embaralhada - normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreino)
    print("sigmoid - batch - embaralhada - normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.fit(bIrisTreinoNaoE)
    print("sigmoid - estocastica - nao embaralhada - normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreinoNaoE)
    print("sigmoid - batch - nao embaralhada - normalizado:%s" %mlpIrisBatch.test(bIrisTeste))
    
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.setFuncaoAtivacao("lRELU")
    mlpIrisBatch.setFuncaoAtivacao("lRELU")
    
    mlpIris.fit(bIrisTreino)
    print("lRelu - estocastica - embaralhada - normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreino)
    print("lRelu - batch - embaralhada - normalizado:%s"%mlpIrisBatch.test(bIrisTeste))
    
    mlpIris.carregarDeArquivo("redes/iris1")
    mlpIrisBatch.carregarDeArquivo("redes/iris1")
    
    mlpIris.setFuncaoAtivacao("lRELU")
    mlpIrisBatch.setFuncaoAtivacao("lRELU")
    
    mlpIris.fit(bIrisTreinoNaoE)
    print("lRelu - estocastica - nao embaralhada - normalizado:%s"%mlpIris.test(bIrisTeste))
    mlpIrisBatch.fit(bIrisTreinoNaoE)
    print("lRelu - batch - nao embaralhada - normalizado:%s"%mlpIrisBatch.test(bIrisTeste))

    #Wine
    mlpWine = MLP([4,3],w,taxa,epocas,tipoFunc="Sigmoid")
    mlpWine.salveEmArquivo("redes/wine")
    mlpWineBatch = MLPBatch([4,3],w,taxa,epocas,tipoFunc="Sigmoid")
    mlpWineBatch.carregarDeArquivo("redes/wine")
    mlpWineNaoEm = MLP([4,3],w,taxa,epocas,tipoFunc="Sigmoid")
    mlpWineNaoEm.carregarDeArquivo("redes/wine")
    mlpWineNaoEmBatch = MLPBatch([4,3],w,taxa,epocas,tipoFunc="Sigmoid")
    mlpWineNaoEmBatch.carregarDeArquivo("redes/wine")
    mlpWine.fit(bWineTreino)
    print("sigmoid - estocastica - embaralhada - Não normalizado:%s"%mlpWine.test(bWineTeste))
   
    mlpWine.carregarDeArquivo("redes/wine")
    bWineTreino.normalizar()
    bWineTeste.normalizar()
    mlpWine.fit(bWineTreino)
    print("sigmoid - estocastica - embaralhada - normalizado:%s"%mlpWine.test(bWineTeste))
    mlpWineBatch.fit(bWineTreino)
    print("sigmoid - batch - embaralhada - normalizado:%s"%mlpWineBatch.test(bWineTeste))
   
    mlpWine.carregarDeArquivo("redes/wine")
    bWineTreinoNaoEmb.normalizar()
    mlpWine.fit(bWineTreinoNaoEmb)
    mlpWineBatch.carregarDeArquivo("redes/wine")
    print("sigmoid - estocastica - não embaralhada - normalizado:%s"%mlpWine.test(bWineTeste))
    mlpWineBatch.fit(bWineTreinoNaoEmb)
    print("sigmoid - batch - não embaralhada - normalizado:%s"%mlpWineBatch.test(bWineTeste))
   
    mlpWine.carregarDeArquivo("redes/wine")
    mlpWine.setFuncaoAtivacao("lRELU")
    mlpWine.fit(bWineTreino)
    mlpWineBatch.carregarDeArquivo("redes/wine")
    mlpWineBatch.setFuncaoAtivacao("lRELU")
    print("lRelu - estocastica - embaralhada - normalizado:%s"%mlpWine.test(bWineTeste))
    mlpWineBatch.fit(bWineTreino)
    print("lRelu  - batch - embaralhada - normalizado:%s"%mlpWineBatch.test(bWineTeste))
   
    mlpWine.carregarDeArquivo("redes/wine")
    mlpWine.setFuncaoAtivacao("lRELU")
    mlpWine.fit(bWineTreinoNaoEmb)
    mlpWineBatch.carregarDeArquivo("redes/wine")
    mlpWineBatch.setFuncaoAtivacao("lRELU")
    print("lRelu - estocastica - não embaralhada - normalizado:%s"%mlpWine.test(bWineTeste))
    mlpWineBatch.fit(bWineTreinoNaoEmb)
    print("Relu  - batch - não embaralhada - normalizado:%s"%mlpWineBatch.test(bWineTeste))
    pass