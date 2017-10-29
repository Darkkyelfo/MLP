from Perceptron import Perceptron
from math import e as euler
from random import random, seed,randint
from numba import jit
class MLP(object):

    #Cria as camadas da rede neural 
    #Recebe um vetor de inteiros onde cada número é a quantidade de neurônios por camcada
    #Recebe um vetor de pesos por camada
    #Receber um vetor de inteiros com a quantidade de iterações por camada
    
    def __init__(self, vNeuronios,pesos,n,qtIteracoes,tipoFunc="sigmoid",a=2):
        self.camadas = [] #Vetor onde cada elemento é um vetor de perceptron.
        self.qtCamadas = len(vNeuronios)
        self.qtIteracoes = qtIteracoes
        self.bias = -1
        self.__setFuncaoAtivacao(tipoFunc)
        self.n = n
        self.a = a
        seed(randint(0,100))#semente para gerar pesos aleatorios
        for i,e in enumerate(vNeuronios):
            self.camadas.append([])
            for j in range(e):
                w = [ random() for k in range(pesos[i]) ]#gera os pesos aleatorios
                print(w)
                p = Perceptron(w,self.n,1,0,self.bias)
                p.setStep(self.ativacao, self.a)
                self.camadas[i].append(p)

    def avaliar(self,atributos):
        self.resultado = []
        return self.__avaliarPorCamada(atributos, 0)
    
    def __avaliarPorCamada(self,entrada,camada):
        self.resultado.append(entrada)
        if(camada == self.qtCamadas):
            return entrada
        
        resultado = []
        for perceptron in self.camadas[camada]:
            resultado.append(perceptron.avaliar(entrada))
        saida = self.__avaliarPorCamada(resultado, camada+1)
        
        return saida
    
    @jit
    def __atualizarPesosSaida(self,classe):
        self.__findDeltasSaida(classe)
        for p,perc in enumerate(self.camadas[-1]):
            #print("perceptron %s" %p)
            for w in range(len(perc.weights)-1):
               # print("w[%s] = %s + %s*%s*%s"%(w,perc.weights[w],self.n,perc.entradas[w],self.delta[0][p]))
                perc.weights[w] = perc.weights[w] + self.n*perc.entradas[w]*self.delta[0][p]
           # print("w[%s] = %s + %s*%s*%s"%(w+1,perc.weights[w+1],self.n,self.bias,self.delta[0][p]))
            perc.weights[-1] = perc.weights[-1] + self.n*self.bias*self.delta[0][p]
            #print(perc.entradas,"saida",perc.weights)  
            #print("\n")
        
    def __findDeltasSaida(self,classe):
        d = []
        for i,perc in enumerate(self.camadas[-1]): #for nos perceptrons da camada de saída
            #print("somatorio:",self.__getDerivada(perc.resultoSomatorio))
            delta = self.__getDerivada(perc.resultoSomatorio)*(classe[i] - perc.saida)
            d.append(delta)
        self.delta.append(d)
            
    def __findDeltaEscondido(self,indDelta,indP,camada):
        d = self.__getDerivada(self.camadas[camada][indP].resultoSomatorio)*self.__somatorioDelta(indDelta,indP,camada)
        return d
    
    def __somatorioDelta(self,indDelta,indP,cam):
        s = 0
        for i,perc in enumerate(self.camadas[cam+1]):
            s = perc.weights[indP]*self.delta[indDelta][i] + s
            #print("s = %s*%s"%(perc.weights[indP],self.delta[indDelta][i]))
        return s
            
    @jit
    def __atualizarPesosOculto(self):
        for indDelta,cam in enumerate(range(self.qtCamadas-2,-1,-1)):
            de = []
            for indP,perc in enumerate(self.camadas[cam]):
                #print(perc.weights)
                delta = self.__findDeltaEscondido(indDelta,indP,cam)
                for w in range(len(perc.weights)-1):
                    #print("w[%s] = %s + %s*%s*%s"%(w,perc.weights[w],self.n,perc.entradas[w],delta))
                    perc.weights[w] = perc.weights[w] + self.n*perc.entradas[w]*delta
                #print("w[%s] = %s + %s*%s*%s"%(w+1,perc.weights[-1],self.n,self.bias,delta))
                perc.weights[-1] = perc.weights[-1] + self.n*self.bias*delta
                de.append(delta)
                #print("\n")
                #print(perc.entradas,"ocultos",perc.weights)
            self.delta.append(de)
    
    @jit
    #Metodo responsavel pro treinar a rede    
    def fit(self,baseTreino):
        for epoca in range(self.qtIteracoes):
            entrou = False
            for index,atr in enumerate(baseTreino.atributos):
                saida = self.avaliar(atr)
                #print("resultados1",self.resultado)
                classe = self.__decimalParaBin(baseTreino.classes[index])
                for k,e in enumerate(saida):
                    if(e!=classe[k]):
                        self.delta = []
                        self.__atualizarPesosSaida(classe)
                        self.__atualizarPesosOculto()
                        entrou = True
                        break
            if(entrou==False):#Caso não ocorram mais erros então para o treinamento
                print(epoca)
                break
                
    
    #define qual vai ser a funcao de ativacao
    #por padrão é a sigmoid
    def __setFuncaoAtivacao(self,tipoFunc="sigmoid"):
        if(tipoFunc=="sigmoid"):
            self.ativacao = "1/(1+e**(-1*x*a))"
            self.dAtivacao = "a*e**(-1*a*x)/((1+e**(-1*a*x))**2)"
        
    def __getDerivada(self,x):
        try:
            x = x
            e = euler
            a = self.a
            return eval(self.dAtivacao)
        except:
            if(x<0):
                return 0
            else:
                return 1
    
    #converte a saida para decimal
    def __binarioParaDecimal(self,binario):
        b = ""
        for i in binario:
            b = b + str(i)
        return int(b,2)
    
    def __decimalParaBin(self,num):
        s = []
        for i in str(bin(num))[2:]:
            s.append(int(i))
        return s
    
 
        
    
