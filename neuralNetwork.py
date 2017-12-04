'''
Created on 4 de nov de 2017
@author: raul
'''
from numba import jit
from copy import deepcopy
from random import random, seed,randint
from sys import float_info
import funcoes
import math
import numpy as np

class Perceptron(object):
    '''
    classe que representa um perceptron 
    '''
    
    def __init__(self,weights,n = 1,qtIteracoes = 10,bias = -1,stepFunction = "linear",parametros=[1,1,1]):
        self.setStep(stepFunction,parametros)
        self.weights = deepcopy(weights) 
        self.n = n
        self.bias = bias
        self.qtIteracoes = qtIteracoes
        self.erros = []
    
    #algumas funcoes tem um valor fixo que as modifica com a sigmoid
    #1/(1+e**(-1*x*a)) geralmente usam esse a como 1
    def setStep(self,funcao,parametros=[1,1,1,1]):
        if(funcao=="linear"):
            self.funcAtivacao = funcoes.Linear()
        elif(funcao=="Sigmoid"):
            self.funcAtivacao = funcoes.Sigmoid(parametros[0])
        elif(funcao=="TanH"):
            self.funcAtivacao = funcoes.TanH()
        elif(funcao=="lRELU"):
            self.funcAtivacao = funcoes.LReLU()
        elif(funcao=="Gaussian"):
            self.funcAtivacao = funcao.Gaussiana()
        else:
            self.funcAtivacao = funcoes.Sigmoid()
    
    def step(self,x):
        return self.funcAtivacao.fx(x)
    
    def dStep(self,x):
        return self.funcAtivacao.dFx(x)
    
    def __somatorio(self,args):
        resultado = 0
        for i,x in enumerate(args):
            resultado = self.weights[i]*float(x) + resultado
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
            for i,atr in enumerate(cTreino.atributos):
                r = self.avaliar(atr)
                if(r != cTreino.classesOri[i]):
                    self.trainFunction(atr, r, cTreino.classesOri[i])
                    erro = erro + 1
            self.erros.append(erro/tBase)

            
    def avaliar(self,atributos):
        self.entradas = [] #guarda a entrada dos dados
        for i in atributos:
            self.entradas.append(i)
        self.saida = self.step(self.__somatorio(atributos)) #guarda a saida encontrada
        
        return self.saida
         
class MLP(object):

    #Cria as camadas da rede neural 
    #Recebe um vetor de inteiros onde cada número é a quantidade de neurônios por camcada
    #Recebe um vetor de pesos por camada
    #Receber um vetor de inteiros com a quantidade de iterações por camada
    
    def __init__(self, vNeuronios,pesos,n,qtIteracoes,tipoFunc="Sigmoid",parametros = [1,1,1,1]):
        self.tipoFunc=tipoFunc
        self.pDaFuncao = parametros
        #--------------------------------------------------------------------------
        self.camadas = [] #Vetor onde cada elemento é um vetor de perceptron.
        self.qtCamadas = len(vNeuronios)
        self.qtIteracoes = qtIteracoes
        self.bias = -1
        self.n = n #taxa de aprendizagem
        seed(randint(0,100))#semente para gerar pesos aleatorios
        for i,e in enumerate(vNeuronios):
            self.camadas.append([])
            for j in range(e):
                w = [ random() for k in range(pesos[i]) ]#gera os pesos aleatorios
                #w = np.array(w,dtype = np.float64)
                #print(w)
                p = Perceptron(w,self.n,1,self.bias,tipoFunc,parametros)
                self.camadas[i].append(p)

    def avaliar(self,atributos):
        self.resultado = []
        return self.definirSaida(self.__avaliarPorCamada(atributos, 0))
    
    def __avaliarPorCamada(self,entrada,camada):
        self.resultado.append(entrada)
        if(camada == self.qtCamadas):
            return entrada
        
        resultado = []
        for perceptron in self.camadas[camada]:
            resultado.append(perceptron.avaliar(entrada))
        saida = self.__avaliarPorCamada(resultado, camada+1)
        
        return saida


    #Metodo responsavel pro treinar a rede    
    def fit(self,baseTreino):
        for epoca in range(self.qtIteracoes):
            entrou = False
            for index,atr in enumerate(baseTreino.atributos):
                saida = self.avaliar(atr)
                classe = self.decimalParaBin(baseTreino.classes[index])
                #print("resultados: ",self.resultado[-1],saida,classe)
                for k,e in enumerate(saida):
                    if(e!=classe[k]):
                        self.delta = []
                        self._findDeltasSaida(classe)
                        self._atualizarPesosSaida()
                        self._findDeltasEscondidos()
                        self._atualizarPesosOcultos()
                        entrou = True
                        break
            if(entrou==False):#Caso não ocorram mais erros então para o treinamento
                print(epoca)
                break
   

    def _atualizarPesosSaida(self):
        for p,perc in enumerate(self.camadas[-1]):
            #print("saida1:",perc.weights)
            for w in range(len(perc.weights)-1):
                perc.weights[w] = perc.weights[w] + self.n*perc.entradas[w]*self.delta[0][p]
            perc.weights[-1] = perc.weights[-1] + self.n*self.bias*self.delta[0][p]
            #print("Saida: ",perc.weights)
        
    def _findDeltasSaida(self,classe):
        d = []
        for i,perc in enumerate(self.camadas[-1]): #for nos perceptrons da camada de saída
            delta = perc.dStep(perc.resultoSomatorio)*(classe[i] - perc.saida)
            d.append(delta)
        self.delta.append(d)
            
    def __findDeltaEscondido(self,indDelta,indP,camada):
        return self.camadas[camada][indP].dStep(self.camadas[camada][indP].resultoSomatorio)*self.__somatorioDelta(indDelta,indP,camada)
    
    def __somatorioDelta(self,indDelta,indP,cam):
        s = 0
        for i,perc in enumerate(self.camadas[cam+1]):
            #print("somatorio:",perc.weights)
            s = perc.weights[indP]*self.delta[indDelta][i] + s
            #print("somatori1: ",s)
        return s

    
    def _atualizarPesosOcultos(self):
        for indDelta,cam in enumerate(range(self.qtCamadas-2,-1,-1)):
            de = []
            for indP,perc in enumerate(self.camadas[cam]):
                for w in range(len(perc.weights)-1):
                    perc.weights[w] = perc.weights[w] + self.n*perc.entradas[w]*self.delta[indDelta-1][indP]
                perc.weights[-1] = perc.weights[-1] + self.n*self.bias*self.delta[indDelta-1][indP]
                #print(perc.weights)
    def _findDeltasEscondidos(self):
        for indDelta,cam in enumerate(range(self.qtCamadas-2,-1,-1)):
            de = []
            for indP,perc in enumerate(self.camadas[cam]):
                de.append(self.__findDeltaEscondido(indDelta,indP,cam))
            self.delta.append(de)
               
    #modifica a funcao de ativacao de todos os perceptrons da rede
    def setFuncaoAtivacao(self,tipoFunc="Sigmoid",parametros=[1,1,1]):
        self.tipoFunc = tipoFunc
        self.pDaFuncao = parametros
        for cam in self.camadas:
            for perc in cam:
                perc.setStep(tipoFunc,parametros)
        
    #converte a saida para decimal
    def __binarioParaDecimal(self,binario):
        b = ""
        for i in binario:
            b = b + str(i)
        return int(b,2)
    
    #retorna um vetor binario do tamanho da quantidade de perceptrons na camada de saida
    #exemplo se houver 3 o vetores sera: entrada(1) -> [0,1,0], entrada(0) -> [1,0,0]
    def decimalParaBin(self,num):
        s = [0]*(len(self.camadas[-1]))
        s[num] = 1
        return s
    
    #funcao que defini qual vai ser a classe de saida da rede
    #na maioria das vezes o vetor na chega a ter somente 1 ou 0
    #essa funcao pega o maior o valor e torna 1 e os demais 0
    def definirSaida(self,saida):
        maior = saida.index(max(saida))
        s = []
        for i,e in enumerate(saida):
            if(i==maior):
                s.append(1)
            else:
                s.append(0)
        return s
    
    #metodo que salva a rede em um arquivo
    def salveEmArquivo(self,caminho,erro=""):
        arq = open(caminho,"w")
        s = str(self.n)+","+str(self.qtIteracoes)+","+str(self.tipoFunc)+","+str(self.bias)+"\n"+str(self.pDaFuncao)+"\n"
        for cam in self.camadas:
            s = s + "#\n" 
            for perc in cam:
                s = s+str(perc.weights)
                s = s.replace("[", "").replace("]", "")
                s = s + "\n"
        s = s + "\nerro%s"%erro
        arq.write(s)
        arq.close()
    
    #carrega uma rede neural salva em arquivo
    def carregarDeArquivo(self,caminho):
        try:
            arq  = open(caminho,"r")
            l = arq.readlines()
            e = l[0].split(",")
            self.n = float(e[0])
            self.qtIteracoes = int(e[1])
            self.tipoFunc = e[2]
            self.bias = float(e[3])
            self.pDaFuncao = list(l[1].replace("\n","").split(","))
            self.pDaFuncao = [float(x) for x in self.pDaFuncao]
            self.camadas = []
            c = -1
            for w in l[2:]:
                if(w=="#\n"):
                    self.camadas.append([])
                    c = c+1
                    continue
                w = list(w.replace("\n","").split(","))
                w = [float(x) for x in w]
                self.camadas[c].append(Perceptron(w,self.n,1,self.bias,self.tipoFunc,self.pDaFuncao))
        except:
            pass
    
    #método que executa a rede neural para uma base e retorn o erro
    def test(self,bTeste):
        erro = 0
        for i,atr in enumerate(bTeste.atributos):
            r = self.avaliar(atr)
            c = self.decimalParaBin(int(bTeste.classes[i]))
            if(r != c):
                erro+=1
        erro = 100*(erro/len(bTeste.classesOri))
        return erro
      
class MLPBatch(MLP):
    
    #@jit
    #Metodo responsavel pro treinar a rede    
    def fit(self,baseTreino):
        for epoca in range(self.qtIteracoes):
            entrou = False
            self.detalAcm = []
            for index,atr in enumerate(baseTreino.atributos):
                saida = self.avaliar(atr)
                classe = self.decimalParaBin(baseTreino.classes[index])
                for k,e in enumerate(saida):
                    if(e!=classe[k]):
                        self.delta = []
                        self._findDeltasSaida(classe)
                        self._findDeltasEscondidos()
                        self._acumularDelta()
                        entrou = True
                        break
            if(entrou==False):#Caso não ocorram mais erros então para o treinamento
                print(epoca)
                break
            self.delta = self.detalAcm
            self._atualizarPesos()

    def _acumularDelta(self):
        if(self.detalAcm==[]):
            self.detalAcm = deepcopy(self.delta)
        else:
            for i,c in enumerate(self.delta):
                for j,delta in enumerate(c):
                    self.detalAcm[i][j] = delta + self.detalAcm[i][j]
    
    def _atualizarPesos(self):
        self._atualizarPesosSaida()
        self._atualizarPesosOcultos()             

class NeuronioRBFsaida(object):
    
    def __init__(self,funcao = "defautl"):
        self._neuEntrada = []
        self._pesos = []
        self.setFunction(funcao)
        
    def addEntrada(self,neuronioInt):
        self._neuEntrada.append(neuronioInt)
        self._pesos.append(1)
    
    def atualizarPeso(self,ind,valor = 1):
        self._pesos[ind]+=valor
        
    def setFunction(self,funcao="defautl"):
        if(funcao == "default"):
            self._funcao = "default"
        else:
            self._funcao = "default"
        
    def avaliar(self):
        
        if(self._funcao=="default"):
            somatorio = 0
            for i,neu in enumerate(self._neuEntrada):
                somatorio+=neu.saida*self._pesos[i]
            sPesos = self._somatorioPesos()
            
            return somatorio/sPesos
        
    def _somatorioPesos(self):
        som = 0
        for i in self._pesos:
            som+=i
        return som
        
         
class NeuronioRBFinter(object):
    
    def __init__(self,centro,raio = math.sqrt(float_info.max-1)):
        self._actvFunction = funcoes.Gaussiana()
        self._c = centro
        self._r = raio
    
    def setCentro(self,c):
        self._c = c
        
    def setRaio(self,r):
        self._r = r
    
    def getCentro(self):
        return self._c
    
    def getRaio(self):
        return self._r
    
    def avaliar(self,x):
        self.saida = self._actvFunction.fx(x,self._c,self._r)
        return self.saida

class RBF(object):
    
    def __init__(self,neuSaida):
        self.neuSaida = neuSaida
        self.camSaida = []
        self.camIntermediaria = []
        for i in range(neuSaida):
            self.camSaida.append(NeuronioRBFsaida())
            self.camIntermediaria.append([])

    def fitDDA(self,bTreino,tetaP = 0.4,tetaN = 0.1 ,valor = 1):
        self._tetaP = tetaP
        self._tetaN = tetaN
        for i,elemento in enumerate(bTreino.atributos):
            deveCriar = True
            for j,neu in enumerate(self.camIntermediaria[bTreino.classes[i]]):
                if(neu.avaliar(elemento)>=self._tetaP):
                    self.camSaida[bTreino.classes[i]].atualizarPeso(j,valor)
                    deveCriar = False
                    break
            if(deveCriar):
                neu = self._gerarNovoNeuronio(elemento, bTreino.classes[i])
                self.camSaida[bTreino.classes[i]].addEntrada(neu)
                distancia = self._calcDistancia(neu.getCentro(),bTreino.classes[i])
                if(distancia>0):
                    nRaio = self._calcNovoRaio(distancia)
                    neu.setRaio(nRaio)
                    self._ajustarRaios(elemento, bTreino.classes[i])      
                
    def avaliar(self,elemento):
        
        resultados = []
        for cam in self.camIntermediaria:
            for neu in cam:
                neu.avaliar(elemento)
        for neu in self.camSaida:
            resultados.append(neu.avaliar())
            
        return self.definirSaida(resultados)
    
    
    def _gerarNovoNeuronio(self,atr,classeCamada):
        nCentro = atr
        neu = NeuronioRBFinter(nCentro)
        self.camIntermediaria[classeCamada].append(neu)
        return neu
        
    def _ajustarRaios(self,atr,classeIgnorada):
        for i,cam in enumerate(self.camIntermediaria):
            if(i!=classeIgnorada):
                for neu in cam:
                    d = np.linalg.norm(np.array(atr)-np.array(neu.getCentro()))
                    r = self._calcNovoRaio(d)
                    neu.setRaio(r)
                    
                                    
    def _calcDistancia(self,centr1,classe):
        try:
            distancias = []
            for i,cam in enumerate(self.camIntermediaria):
                if(i != classe):
                    for neu in cam:
                        d = np.linalg.norm(np.array(centr1)-np.array(neu.getCentro()))
                        distancias.append(d)
            distancias.sort()
            return distancias[0]
        except:
            return 0
    
    def _calcNovoRaio(self,dist):
        return math.sqrt((-1*dist)/math.log(self._tetaN,math.e))

    def definirSaida(self,saida):
        maior = saida.index(max(saida))
        s = []
        for i,e in enumerate(saida):
            if(i==maior):
                s.append(1)
            else:
                s.append(0)
        return s
    
    #retorna um vetor binario do tamanho da quantidade de perceptrons na camada de saida
    #exemplo se houver 3 o vetores sera: entrada(1) -> [0,1,0], entrada(0) -> [1,0,0]
    def decimalParaBin(self,num):
        s = [0]*(len(self.camSaida))
        s[num] = 1
        return s
    
    def testar(self,bTeste):
        erro = 0
        for i,atr in enumerate(bTeste.atributos):
            if(self.avaliar(atr) != self.decimalParaBin(bTeste.classes[i])):
                erro+=1
        return erro/len(bTeste.classes)
                
            
            

