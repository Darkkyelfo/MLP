'''
Created on Aug 14, 2017

@author: raul
'''

from numba import jit
class SepararBase(object):
    
    @staticmethod
    def ehNumero(numero):#teste se a string é um número
        t = numero.replace(".","",1)
        t = t.replace("\n","")
        t = t.replace(" ","")
        t = t.replace("-","",1)
        if(t.isdigit()):
            return True
        return False
        
    @staticmethod
    @jit
    def coletarDadosNumericos(dados,classes,ignorar,separador=","):
        classe = []
        atributos = []
        for i in dados:
            cla = []
            atr = []
            deveAdd = True
            temp = i.split(separador)
            aux = []
            for k in temp:
                if("" != k):
                    aux.append(k.replace("\n",""))
            for index, e in enumerate(aux):
                if(index in classes):
                    cla.append(e.replace("\n",""))
                elif (index in ignorar):
                    continue
                else:
                    if(SepararBase.ehNumero(e)):
                        atr.append(float(e))
                    else:
                        deveAdd = False
                        break
            if(deveAdd):
                if(SepararBase.ehNumero(cla[0])):
                    classe.append(int(cla[0]))
                else:
                    classe.append(cla[0])
                atributos.append(atr)
        return classe,atributos
                     
                    
            

        