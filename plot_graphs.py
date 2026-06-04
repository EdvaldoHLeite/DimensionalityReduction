from train_test import carregar, criar_pasta
from load_datasets import *
from sklearn.feature_selection import mutual_info_classif
#from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# nomes das bases para plotagem de graficos
from load_datasets import obs_network, letter, user_knowledge, mice, wine_quality_red, wine_quality_white, waveform


def info_gain_graph(base, repeticoes):
    nome = base[2]
    colunas = base[3]

    # data frame
    feat_importances = pd.Series(data=np.arange(len(colunas)), index=colunas)
    # calculando media
    for i in range(repeticoes):
        data = pd.read_csv("resultados/repeticoes-"+str(repeticoes)+
                            "/"+nome+"/reducoes_resultados/info_gain-"+
                            str(i+1)+".csv")

        data.columns = ['columns', 'values']
        dataSeries = pd.Series(data=np.array(data['values']), index=np.array(data['columns']).astype(str))
        #print(dataSeries.index)
        for coluna in list(colunas):
            #print(data[coluna])
            feat_importances[coluna] += dataSeries[coluna]/repeticoes
    # plotagem
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind='barh', color='teal')
    plt.title(nome+" - Info Gain")
    plt.show()
    plt.close()

def fisher_score_graph(base, repeticoes):
    nome = base[2]
    colunas = base[3]

    # data frame
    feat_importances = pd.Series(data=np.arange(len(colunas)), index=colunas)
    # calculando media
    for i in range(repeticoes):
        data = pd.read_csv("resultados/repeticoes-"+str(repeticoes)+
                            "/"+nome+"/reducoes_resultados/fisher_score-"+
                            str(i+1)+".csv")

        data.columns = ['columns', 'values']
        dataSeries = pd.Series(data=np.array(data['values']), index=np.array(data['columns']).astype(str))
        #print(dataSeries.index)
        for coluna in list(colunas):
            #print(data[coluna])
            feat_importances[coluna] += dataSeries[coluna]/repeticoes
    # plotagem
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind='barh', color='teal')
    plt.title(nome+" - Fisher`s Score")
    plt.show()
    plt.close()
    
def fisher_info_gain():
    bases = [banknote(), climate(), debrecen(),
        pima(), vcolumn(), wdbc(), spambase(),
        occupancy()]
    #bases=[debrecen()]
    for base in bases:
        fisher_score_graph(base, 100)
        info_gain_graph(base, 100)

    
def graphs():
    ################ PCAs #################################################################
    nomes_reducao = [
                     'chi2_square',
                     #'LASSO',
                     'fishers_score',
                     'info_gain',
                     'Forward',
                     'RFE',
                     #'variance_threshold'
                     ]
    
    '''arranjos = []
    for reducao in nomes_reducao:
        arranjos.append(["PCA", reducao])'''

    pcaPrincipal = "MCEPCA"
    '''arranjos = [
        [pcaPrincipal, 'Forward'],
        [pcaPrincipal, 'info_gain'],
        [pcaPrincipal, 'chi2_square']
        ]'''
    arranjos = [[pcaPrincipal, 'Forward', 'info_gain', 'chi2_square']]
    #print(arranjos)
    
    ############## configuracoes das imagens graficas ######################################
    config = {"PCA":['PCA', 'b', '.'],
              "MCEPCA":["MCEPCA", 'b', '.'],
              "info_gain":["info_gain", 'k', '+'],
              #"fishers_score":["fishers_score", 'g', '.'],
              #"correlation_coefficient":['correlation_coefficient', 'g', '.'],
              "chi2_square":["chi2_square", 'y', '*'],
              #"RFE":["RFE", 'r', '.'],
              "Forward":["Forward", 'r', '2'],
              #"variance_threshold":["Variance Threshold: 0.1", 'g', '.'],
              #"LASSO":["LASSO", "r", "."]
              }

    ###### classificadores    
    classificadores = ['tree','knn', 'gnb', 'lda']


    ####### bases
    bases2 = ["Banknote", "Climate", "Debrecen", 
             "Pima", "VColumn", "WDBC", "Occupancy"]
    bases3 = [
        #obs_network(),
        letter()[2],
        user_knowledge()[2],
        mice()[2],
        wine_quality_red()[2],
        wine_quality_white()[2],
        waveform()[2]
    ]

    bases = bases2
    ##### executando para as configuracoes
    for base in bases:
        for arranjo in arranjos:
            carregar("resultados/repeticoes-100",
                     base,
                     classificadores,
                     arranjo,
                     config)
graphs()
#fisher_info_gain()       


