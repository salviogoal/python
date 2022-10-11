# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:54:19 2022

Sálvio Gomes de Almeida e Thieres Nardy Dias

"""

#Técnica Filtro Mutual Information (MI) para seleção de atributos
from sklearn.feature_selection import mutual_info_classif as mic
#método de cross-validation
from sklearn.model_selection import cross_val_score
#classificador Naive Bayes para modelos multinomiais - adequado para atributos com valores discretos
from sklearn.naive_bayes import MultinomialNB

#bibliotecas padrão Python
#n-dimensional array (numéricos)
import numpy as np
#gerar números aleatórios
import random as rd

from solution import Solution

#número de folds (inner cross-validation in Wrapper)
K_FOLDS = 5
#valor muito pequeno igual a 0.0000000001
e = 10 ** (-10)
#constantes que indexam o mapeamento (feature -> score) do Ranking MI
FEATURE = 0
SCORE = 1

#função para avaliar o critério de relevância entre duas soluções 
#(duas avaliações Wrapper)
#verificar se s2 possui relevância sobre s1
def relevance_criterion( s2, s1, k ):
    
    count = 0
    
    #acurácia média de s2 é maior que a de s1
    #então s2 possui, em média, melhor acurácia em relação a s1
    if( s2.get_average_acc() > s1.get_average_acc() ):
        
        #contar em quantos folds s2 teve melhor acurácia que s1
        for i in range( 0, K_FOLDS ):
            
            #se acurácia de s2 for melhor que a de s1
            if( s2.get_acc()[i] > s1.get_acc()[i] ):
                #contagem de relevância
                count += 1
        
        #se a quantidade de folds que a acurácia de s2 for melhor que s1 for maior ou igual a k
        #então s2 possui relevância sobre s1        
        if( count >= k ):
            return True

    #s2 não possui relevância sobre s1    
    return False

#método da roleta, ou seleção probabilística
def roulette_wheel( ranking, probSel, sum_scores, select_features, N_FEATURES  ):
    
    #guarda o resultado da rodada da roleta
    roulette_round = 0.0
    #índice do atributo
    feature = 0
    
    #selecionar N atributos através da roleta probabilística
    for d_feature in range( 0, N_FEATURES ):
        
        #executa a roleta com distribuição uniforme
        roulette_round = rd.uniform(0, 1)
        
        #de 0 até N-1 (número total de atributos da base)
        index_ranking = 0
        
        while ( ( roulette_round > 0 ) and ( index_ranking < len( select_features ) ) ):
            
            #recebe o índice do atributo f do ranking
            feature = ranking[index_ranking][FEATURE]
            
            #se atributo não selecionado
            if ( not select_features[feature] ):
                
                #retira a probabilidade de seleção do atributo desta rodada da roleta
                #para verificar se o atributo foi o sorteado na rodada
                roulette_round -= probSel[feature]
                
                #se roleta parou no atributo
                if( roulette_round < 0 ):
                    
                    #seleciona atributo
                    #seleção sem reposição
                    select_features[feature] = 1
                    
            #próximo atributo
            index_ranking += 1
        
        #zera soma dos scores
        sum_scores = 0.0
        #zera todas as probabilidades de seleção
        probSel = [0.0 for p in probSel]
        
        #recalcular o somatório do score 
        #considerar apenas os atributos que ainda não foram selecionados pela roleta
        for feature_score in ranking:
            
            #recebe o índice do atributo
            feature = feature_score[FEATURE]
            
            #se atributo não selecionado
            if( not select_features[feature] ):
                
                #somatório do score de todos os atributos
                sum_scores += feature_score[SCORE]
                
        #recalcular as probabilidades de selecionar cada atributo individualmente
        #considerar apenas os atributos que ainda não foram selecionados pela roleta
        for feature_score in ranking:
            
            #recebe o índice do atributo
            feature = feature_score[FEATURE]
            
            #se atributo não selecionado
            if( not select_features[feature] ):
                
                #calcula a probabilidade de seleção de cada atributo
                probSel[ feature ] = feature_score[SCORE] / sum_scores
            
    #retorna os atributos selecionados pela roleta
    return select_features
    

#metaheurística GRASP (Greedy Randomized Adaptive Search Procedure)
def search( X, y, RESTART_GRASP, N_FEATURES ):
    
    #fornece a semente para geração de números aleatórios
    #reproduzibilidade dos números pseudo-aleatórios
    rd.seed( 1 )
    
    #primeiro é o atributo zero
    index_feature = 0
    #guarda a soma do score Mutual Information (MI) de todos os atributos
    sum_scores = 0.0
    #dicionário que armazena o mapeamento (índice atributo -> score MI)
    #Ranking dos atributos pela métrica Filtro MI
    ranking = {}
    
    """
    Aplicação da etapa Filtro para seleção de atributos.
    """
    #aplica a técnica Filtro Mutual Information (MI) para cada atributo da base
    #considera todos os atributos discretos
    mic_scores = mic( X, y, discrete_features = True, random_state = 1)
    
    #para cada score MI de cada atributo (na ordem dos atributos da base) 
    for score in mic_scores:
        
        #se score MI negativo, score MI do atributo deve ser 0.0
        if( score < 0 ):
            #garantir um score não negativo para MI
            score = 0.0
        
        #armazena o mapeamento índice do atributo -> score MI
        #soma o score do atributo com um valor muito pequeno para não termos score igual a 0
        ranking[index_feature] = score + e
        #somatório do score de todos os atributos
        sum_scores += ranking[index_feature]
        #recupera o score do próximo atributo
        index_feature += 1
    
    #ordena de forma decrescente o mapeamento índice atributo -> score MI, pelo valor do score MI 
    #do maior score MI para o menor
    #criação do ranking dos atributos pelo score da métrica MI
    ranking = sorted( ranking.items(), key=lambda x:x[1],  reverse = True )
    
    """
    Final da etapa Filtro.
    """
    
    #contagem de restarts do GRASP
    restart = 0
    #cria um array para armazenar as probabilidades de seleção para cada atributo
    probSel = np.zeros( len( ranking ), dtype=float )
    
    #criação do classificador Naive Bayes Multinominal
    clf = MultinomialNB()
    
    #melhor solução de todas encontradas pelo GRASP
    best = Solution()
    
    #Reinícios do GRASP com componentes aleatórios
    while restart < RESTART_GRASP:
        
        #k é gerado aleatoriamente no intervalo de [2,4] em cada novo restart do GRASP
        #k se refere a um critério de relevância
        k = rd.randint( 2, 4 )
            
        #cálculo das probabilidades de selecionar cada atributo individualmente
        #método de seleção probabilística - roleta
        for feature_score in ranking:
            
            feature = feature_score[FEATURE]
            
            #calcula a probabilidade de seleção de cada atributo
            probSel[feature] = feature_score[SCORE] / sum_scores
        
        #bitset que indica se um dado atributo foi selecionado pelo método da roleta
        #inicialmente nenhum atributo é selecionado
        select_features = np.zeros( len( ranking ), dtype=int )
    
        #método da roleta, seleciona N_FEATURES 
        select_features = roulette_wheel( ranking, probSel, sum_scores, select_features, N_FEATURES )
        
        #ranking (conforme MI) dos atributos selecionados pelo método da roleta
        rank_roulette_choose = np.zeros( N_FEATURES, dtype=int )
        
        #iterar sobre os atributos selecionados pela roleta
        index_rank = 0
        
        #para cada atributo do ranking MI
        #ranquear os atributos selecionados pela roleta
        for feature_score in ranking:
            
            #recebe o índice do atributo
            feature = feature_score[FEATURE]
            
            #se atributo selecionado pela roleta
            if( select_features[feature] ):
                
                #criação do ranking dos atributos selecionados pelo método roleta
                rank_roulette_choose[index_rank] = feature
                index_rank += 1
        
        """
        Aplicação da etapa Wrapper para seleção de atributos.
        - IWSS - Incremental Wrapper Subset Selection
        """
        
        #soluções do problema para cada restart do GRASP
        solution1 = Solution()
        solution2 = Solution()
        
        #cria uma lista para adicionar os atributos de forma incremental
        incremental = []
        
        #adiciona o primeiro atributo do ranking no passo incremental
        incremental.append( rank_roulette_choose[0] )
        
        #avaliação Wrapper do primeiro atributo do ranking
        #técnica Wrapper usando o classificador Naive Bayes Multinominal
        acc = cross_val_score( clf, X[0:,incremental], y, cv = K_FOLDS )
        
        #adiciona a solução inicial 
        solution1.add_solution( incremental, acc )
        
        #tenta adicionar de forma incremental os N_FEATURES - 1 atributos restantes
        #realiza avaliação Wrapper para cada tentativa incremental
        for i in range( 1, N_FEATURES ):
            
            #adiciona o próximo atributo do ranking
            incremental.append( rank_roulette_choose[i] )
            
            #avaliação Wrapper do próximo atributo do ranking adicionado
            #técnica Wrapper usando o classificador Naive Bayes Multinominal
            acc2 = cross_val_score( clf, X[0:,incremental], y, cv = K_FOLDS )
            
            #adiciona nova solução de forma incremental 
            solution2.add_solution( incremental, acc2 )
            
            #avaliação do critério de relevância entre duas soluções
            #solution2 possui relevância sobre solution1
            if( relevance_criterion( solution2, solution1, k ) ):
                
                #houve relevância da nova solução criada pelo passo incremental
                solution1.add_solution( solution2.get_subset(), solution2.get_acc() )
                
            else:
                
                #retira o atributo que tentou-se adicionar à solução
                incremental.pop()
                #apaga solution2
                solution2 = Solution()
        
        #se solution1 tiver relevância sobre a melhor solução (best) encontrada pelo GRASP
        if( relevance_criterion( solution1, best, k ) ):
            
            #então a nova melhor solução encontrada pelo GRASP é best
            best.add_solution( solution1.get_subset(), solution1.get_acc() )
         
        #incrementa restart do GRASP
        restart += 1
        
    #retorna a melhor solução encontrada pelo GRASP
    return best