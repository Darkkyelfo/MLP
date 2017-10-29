'''
Created on 15 de out de 2017

@author: raul
'''
from SepararBase import SepararBase
from Base import Base
class CreateBaseFromFile(object):
    '''
    classdocs
    '''

    @staticmethod
    def createFromFile(arquivo,classes,ignorar,startFromLine = 0,separador = ","):
        arq = open(arquivo,"r")
        dados = arq.readlines()
        classes,atributos = SepararBase.coletarDadosNumericos(dados[startFromLine:], classes, ignorar,separador)
        arq.close()
        return Base(classes,atributos)
        