# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:09:55 2022

Sálvio Gomes de Almeida e Thieres Nardy Dias

"""
import numpy as np

#classe que representa uma solução do problema de seleção de subconjuntos de atributos
class Solution:
    
    #construtor da classe Solução
    def __init__( self ):
        
        #número de folds (inner cross-validation in Wrapper)
        self.K_FOLDS = 5
        #cria uma lista de K_FOLDS acurácias com 0.0
        self.acc = [0.0] * self.K_FOLDS
        #subconjunto de atributos inicialmente vazio
        self.subset = []

    #adiciona uma nova solução
    def add_solution( self, subset, acc  ):
        
        #realiza cópias dos parâmetros acc e subset
        #adiciona o subconjunto de atributos
        self.subset = subset.copy()
        #adiciona a lista de K_FOLDS acurácias
        self.acc = acc.copy()
        
    #retorna a acurácia média da solução
    def get_average_acc( self ):
        
        #calcula a média da lista de acurácias
        return np.mean( self.acc )
    
    #retorna o subconjunto de atributos da solução
    def get_subset( self ):
        
        return self.subset
    
    #retorna a lista de acurácias
    def get_acc( self ):
        
        return self.acc
    
    #impressão da solução
    def print_solution( self ):
        
        print( self.subset )
        print( self.acc )
        print( np.mean( self.acc ) )
        