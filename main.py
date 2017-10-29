'''
Created on 25 de out de 2017

@author: raul
'''
from MLP import MLP
from Base import Base
if __name__ == '__main__':
    #Teste Com o xor
    baseXor = Base([0,1,1,0],[[0,0],[0,1],[1,0],[1,1]])
    #baseXor = Base([0],[[1,1]])
    #w = [np.random.uniform(low=0.1, high=1, size=(3,)),np.random.uniform(low=0.1, high=1, size=(3,))]
    w = [3,3]
    mlp = MLP([2,1],w,0.5,100000)
    mlp.fit(baseXor)
    print(mlp.avaliar([0,0]))
    #print(mlp.resultado)
    print(mlp.avaliar([0,1]))
    #print(mlp.resultado)
    print(mlp.avaliar([1,0]))
    #print(mlp.resultado)
    print(mlp.avaliar([1,1]))
   # print(mlp.resultado)
    pass