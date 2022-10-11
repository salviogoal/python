# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:58:29 2022

Sálvio Gomes de Almeida e Thieres Nardy Dias

"""

#bibliotecas de pré-processamento de dados
#codificar o atributo classe nominal como um número
from sklearn.preprocessing import LabelEncoder
#discretizar atributos contínuos
from sklearn.preprocessing import KBinsDiscretizer

#biblioteca para o carregamento dos conjuntos de dados em formato ARFF
from scipy.io import arff

#bibliotecas padrão Python
import pandas as pd

#função de pré-processamento para realização de Feature Subset Selection
def preprocessing_FSS( dataset ):

    #carrega o dataset em formato Attribute-Relation File Format (ARFF)
    #dados e seu metadados
    data, metadata = arff.loadarff( f'datasets\{dataset}' )
    
    #imprime o nome do dataset carregado
    print( f"Carregamento do conjunto de dados: {metadata.name.upper()}." )
    
    #converte os dados em um dataframe Pandas
    df_data = pd.DataFrame( data )
    
    #se o atributo classe for nominal
    if( metadata.types()[-1].upper() == "NOMINAL" ):
        
        #encoder para o atributo classe (nominal to number)
        le = LabelEncoder()
        
        #codifica o atributo classe em representação numérica
        df_data.iloc[0:,-1] = le.fit_transform( df_data.iloc[0:,-1] )
     
    #converte o atributo classe para nd-array 
    y = df_data.iloc[0:,-1].to_numpy( dtype=int )

    #discretizador de atributos contínuos
    kbd = KBinsDiscretizer( n_bins=5, encode='ordinal', strategy='uniform' )
    #realiza o pré-processamento - discretização dos atributos preditores
    X_transform = kbd.fit_transform( df_data.iloc[0:,0:-1], y ) 
    #converte para inteiros
    X_transform = X_transform.astype( int )
    
    #retorna a matriz dos atributos preditores e o vetor do atributo classe
    return X_transform, y
