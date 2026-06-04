import numpy as np
import pandas as pd
from numpy import linalg as eigen # autovetores e autovalores
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2, SelectKBest # chi square
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler

def PCA(X): # retorna matriz centralizada e covariancia
    media_matriz = X.mean(axis=0)

    X = X - media_matriz 
    XT = X.T #X.transpose() # transposta de X ou X.T

    covariancia = XT @ X #np.dot(XT, X) # X.T @ X
    covariancia = covariancia/len(X) 
    aut_val, aut_vet = eigen.eig(covariancia) # (np.float(_) for _ in )

    # convertendo resultado de complexo para real
    aut_val = np.real(aut_val)
    aut_vet = np.real(aut_vet)

    #aut_val, aut_vet = eigen.eig(covariancia.T) # uma matriz e sua transposta tem os mesmos autovetores
    # os autovetores são as colunas
    save = str(aut_val)
    #ordenado = np.argsort(aut_val)[::-1]
    save2 = str(aut_val)
    if save != save2:
        print("Opa os autovetores foram modificados durante o PCA, não podem ser usados no MCEPCA")
    
    '''Ed = aut_vet[:, ordenado[:d]]
    aut_vet = Ed
    aut_val = aut_val[ordenado[:d]]'''
    #return Ed, aut_val, aut_vet

    return aut_vet, aut_val

def info_gain(X, y, dataframe_columns):
    # valor mutual entre as features
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, dataframe_columns) # ganho de informação
    
    # indices features ordenados pelo maior ganho de informação
    ordenado = np.argsort(feat_importances)[::-1]
    return list(ordenado), feat_importances

def fishers_score(X, y, dataframe_columns):
    ranks = fisher_score.fisher_score(X, y)
    feat_importances = pd.Series(ranks, dataframe_columns)
    ordenado = np.argsort(feat_importances[::-1]) # ordenado em ordem decrescente
    return list(ordenado), feat_importances

# escolhe as features por menor soma de coefientes
def correlation_coefficient(data_frame):
    '''correlacao = data_frame.corr()
    a=abs(correlacao[target])
    result=a[a<threshold]    
    return list(result.index)'''
    
    correlacao = data_frame.corr() # matriz de correlacoes
    somas_corr = [] # somas das correlacoes para cada feature
    for c in correlacao.columns:
        somas_corr.append(correlacao[c].sum())
        
    somas_corr = np.array(somas_corr)
    ordem = np.argsort(somas_corr)
    
    return list(ordem)

# variacao selecionando o melhor grupo para cada quantidade de features
def correlation_coefficient_2(data_frame):    
    correlacao = data_frame.corr() # matriz de correlacoes
    
    # combinacoes para cada k quantidade de features    
    grupos_indices = []
    # inicia com o maior valor possivel de somas de coeficientes
    menor_soma = (1.0)*len(correlacao)
    # para cada quantidade de features
    for k in range(len(correlacao)):
        # para cada feature como target, um grupo diferente
        melhor_grupo = []
        for ind_feat in range(len(correlacao)):
            ind_menores = np.argsort(correlacao[ind_feat])[k+1]
            soma = ind_menores.sum()
            
            if soma < menor_soma:
                menor_soma = soma
                melhor_grupo = ind_menores
            
        grupos_indices.append(melhor_grupo)
        
    return grupos_indices

def selecionar_indices_chi2(chi2_features):
    # selecionando os indices que são proeminentes
    indices = []
    indices_bool = chi2_features.get_support()
    for i in range(len(indices_bool)):
        if indices_bool[i]:
            indices.append(i)
    return indices
def chi2_square(X, y, k):
    # passando para categorico, tratando casos negativos
    X_cat = X.astype('int')
    X_cat_pos = np.arange(len(X_cat)*len(X_cat[0])).reshape(len(X_cat), len(X_cat[0]))
    unicos = np.unique(X_cat) # diferentes valores inteiros
    for unico_ind in range(len(unicos)):
        unico = unicos[unico_ind]
        X_cat_pos[X_cat==unico] = unico_ind + 1
    
    ordem = [] # indices ordenados
    for ki in range(k): # a medida que ki aumenta surge um novo indice
        chi2_features = SelectKBest(chi2, k=ki+1)
        X_kbest_features = chi2_features.fit_transform(X_cat_pos, y)
        
        # selecionando os indices que são proeminentes
        indices = selecionar_indices_chi2(chi2_features)   
        # adiciona o novo indice aos indices ordenados
        #print(len(np.setdiff1d(ordem, indices)))
        #ordem.append(np.setdiff1d(ordem, indices))
        for indice in indices:
            if indice not in ordem:
                ordem.append(indice)
                break

    return ordem

def forward_linear_regression(X, y):
    #dataset = pd.DataFrame(data=y, columns=['target'])
    
    #### passando o y para numerico
    '''f=0 # numero de um target
    valores = y.unique()
    for val in valores:
        f=f+1
        # substitui cada target por um numero
        dataset['target'].replace([feature], [f])'''
    # diferentes valores de y
    '''valores = np.unique(y)
    # troca um target por um numero
    dataset['target'].replace(list(valores), list(np.arange(len(valores))))
    y = np.array(dataset['target'])'''
    
    lr = LinearRegression()
    
    ordem = [] 
    # encontra as features para cada quantidade 
    # adicionando em seguida na lista ordenada, as primeiras que aparecem
    for k in range(len(X[0])):
        ffs = SequentialFeatureSelector(estimator=lr, k_features=k+1)
        ffs.fit(X, y)
        indices = ffs.k_feature_idx_ #list(np.where(rfe.support_ == True)[0])
        # verifica se o indice nao existe na lista ordenada
        for indice in indices:
            if indice not in ordem:
                ordem.append(indice)
                break
        
    return ordem

def RFE_linear_regression(X, y):
    dataset = pd.DataFrame(data=y, columns=['target'])
    
    #### passando o y para numerico
    '''f=0 # numero de um target
    valores = y.unique()
    for val in valores:
        f=f+1
        # substitui cada target por um numero
        dataset['target'].replace([feature], [f])'''
    # diferentes valores de y
    valores = np.unique(y)
    # troca um target por um numero
    dataset['target'].replace(list(valores), list(np.arange(len(valores))))
    y = np.array(dataset['target'])
    
    lr = LinearRegression()
    
    ordem = [] 
    # encontra as features para cada quantidade 
    # adicionando em seguida na lista ordenada, as primeiras que aparecem
    for k in range(len(X[0])):
        rfe = RFE(estimator=lr, n_features_to_select=k+1, step=1)
        #print("Y no RFE", y)
        rfe.fit(X, y)
        indices = list(np.where(rfe.support_ == True)[0])
        # verifica se o indice nao existe na lista ordenada
        for indice in indices:
            if indice not in ordem:
                ordem.append(indice)
                break
        
    return ordem

def variance_threshold(X, limiar):
    # limiar sendo 0: a variancia das features devem ser diferente de zero
    select_feature = VarianceThreshold(threshold=limiar)
    
    # quando não encontra variancia do limiar passado
    try:
        select_feature.fit(X)
    except ValueError:
        return variance_threshold(X, limiar-0.1)
    
    # array com os indices das features selecionadas
    features = []
    # array com True se a variancia for diferente
    variancia_diferente = select_feature.get_support()
    #print(select_feature.variances_)
    
    for i in range(len(X[0])):
        if variancia_diferente[i]:
            features.append(i)
            
    return features
            
def LASSO(X, y):
    logistic = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=7).fit(X,y)
    model=SelectFromModel(logistic, prefit=True)
    
    # array com os indices das features selecionadas
    features = []
    # array com True se a variancia for diferente
    variancia_diferente = model.get_support()


    #model2 = make_pipeline(StandardScaler(), Lasso(alpha=.015))
    '''model2 = Lasso(alpha=1)
    model2.fit(X, y)
    a = abs(np.array([x for x in model2.coef_]))
    
    order_features = a.argsort()[(-1)*len(a):][::-1]
    return order_features'''

    #print(variancia_diferente)
    #print(np.std(X, 0) * logistic.coef_)
    for i in range(len(X[0])):
        if variancia_diferente[i]:
            features.append(i)
            
    return features

def MCEPCA(W, X, k, classes, Y, autovalores, autovetores):
    n = len(X)
    d = len(X[0])
         
    # media das features
    W_mean = [[0 for x in range(d)] for y in range(2)]
    for c in range(2): # classes
        for i in range(d): # features
            numerador = 0
            denominador = 0
            for j in range(n): # pontos
                numerador += W[j][i] * int(Y[j] == classes[c])  # wji eh o transposto de wij
                denominador += int(Y[j] == classes[c])
            W_mean[c][i] = numerador/denominador        
    
    # score das features
    score = []
    for i in range(d):
        if autovalores[i] != 0:
            score.append((W_mean[0][i] - W_mean[1][i])**2/autovalores[i])
        else:
            score.append(0)

    ordenado = np.argsort(score)[::-1] # vetor com os indices de score ordenados
    Sk = autovetores[:, ordenado[:k]] # seleciona as k colunas
    return Sk






        
    
