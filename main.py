# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:54:19 2022

Sálvio Gomes de Almeida e Thieres Nardy Dias

"""

#bibliotecas de Aprendizado de Máquina (sklearn)
#método k-fold estratificado (particionamento da base de dados em k-folds)
from sklearn.model_selection import StratifiedKFold
#classificador Naive Bayes para modelos multinomiais - adequado para atributos com valores discretos
#from sklearn.naive_bayes import MultinomialNB
#classificador KNN - K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier


#bibliotecas do projeto PUC-MG
#importa o pacote da metaheurística GRASP (Greedy Randomized Adaptive Search Procedure)
import grasp
#importa a biblioteca para carregar e pré-processar os datasets
import load_transform_datasets as load

#número de folds (outer cross-validation)
K_FOLDS = 10
#número de reínicios do GRASP
RESTART_GRASP = 50
#número de atributos para realizar amostragem na etapa da roleta
N_FEATURES = 100

#datasets considerados no projeto PUC-MG
DATASETS = ["DLBCL.arff"]

for dataset in DATASETS:
    
    #impressão do ínicio do processo de seleção de atributos
    print( f"Feature Subset Selection in Dataset {dataset}.")

    #X se refere ao espaço matricial das features (atributos preditores)
    #y se refere ao espaço vetorial do atributo classe
    X, y = load.preprocessing_FSS( dataset )
    
    #executa o método k-fold estratificado (refletir a proporcionalidade entre as classes em cada fold)
    #com número de partições igual a n_splits = K_FOLDS
    #com o embaralhamento dos exemplos da base a partir da semente de randomicidade 1
    skf = StratifiedKFold( n_splits = K_FOLDS, shuffle = True, random_state = 1 )
    
    #criação do classificador Naive Bayes Multinominal
    #clf = MultinomialNB()
    
    #criação do classificador KNN
    clf = KNeighborsClassifier( n_neighbors=5 )
    
    
    #média da acurácia final e da cardinalidade do subconjunto selecionado
    sum_acc = 0.0
    sum_subset = 0
    
    #realiza o split da base de dados em treinamento e teste de acordo com o método k-fold estratificado 
    #recupera a lista contendo os índices de treinamento e teste
    for train_index, test_index in skf.split(X, y):
        
        #recupera as partições de treinamento e teste das atributos preditores
        X_train, X_test = X[train_index], X[test_index]
        #recupera as partições de treinamento e teste do atributo classe
        y_train, y_test = y[train_index], y[test_index]
        
        #realiza a busca pelo GRASP do subconjunto de atributos
        #com parâmetros:
        #número de reinícios do GRASP 
        #número de atributos para amostrar pelo método da roleta
        solution = grasp.search( X_train, y_train, RESTART_GRASP, N_FEATURES )
        
        #treinamento de um modelo de classificação usando o subconjunto de atributos selecionado pelo GRASP
        #Naive Bayes Multinominal
        #uso da dobra de treinamento
        model = clf.fit( X_train[0:,solution.get_subset()], y_train )
        
        #teste do modelo construído com o subconjunto de atributos selecionado GRASP a partir da dobra de teste
        scores = model.score( X_test[0:,solution.get_subset()], y_test )
        
        #somatório das acurácias encontradas e da cardinalidade do subconjunto de atributos selecionado
        sum_acc += scores
        sum_subset += len( solution.get_subset( ) )
        
        #resultados da rodada de cross-validation
        print( "Acurácia média da rodada: " + str( scores ) )
        print( "Número de atributos selecionados na rodada: " + str( len( solution.get_subset( ) ) ) )
        
    print( "Acurácia média final: " + str( sum_acc / K_FOLDS ) )
    print( "Média do número de atributos selecionados: " + str( sum_subset / K_FOLDS ) )    