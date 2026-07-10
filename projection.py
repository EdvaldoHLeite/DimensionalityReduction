
""" Recebe os reducao, e as bases, para fazer a projecao """

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import train_test_split

from train_test import treino_teste, treino_teste_unico, salvar
from feature_selection import *

import os
from train_test import criar_pasta

# classificadores
knn_reducao = knn(n_neighbors=1)
gnb_reducao = gnb()
tree_reducao = tree()
lda_reducao = lda()

def project_datasets(bases, nomes_reducao, numero_repeticoes):
    
    # para cada base
    for indice in range(len(bases)):
        # dados da base
        X = bases[indice][0]
        y = bases[indice][1]
        nome_base = bases[indice][2]
        nomes_colunas = bases[indice][3]
        # print("Dataset", len(bases[indice]))
        
        # divisão das n iteracoes, para serem salvas depois
        # Todos os testes, iteracoes, classificadores e metodos devem usar a mesma divisao de treino e teste, ao menos para cada base
        df_treino_indices = {}
        df_teste_indices = {}
        pasta_indices = "bases/"+nome_base+"/dividir_treino_teste"
        criar_pasta(pasta_indices)
        if os.path.isfile(pasta_indices+"/treino-"+str(numero_repeticoes)+".csv") and os.path.isfile(pasta_indices+"/teste-"+str(numero_repeticoes)+".csv"):
            df_treino_indices = pd.read_csv(pasta_indices+"/treino-"+str(numero_repeticoes)+".csv")
            df_teste_indices = pd.read_csv(pasta_indices+"/teste-"+str(numero_repeticoes)+".csv")
        else:
            # ao inves de dividir a matriz, sera dividido os indices das linhas (gambiarra)
            X_indices = np.arange(len(X))
            for iteracao in range(numero_repeticoes):
                # uso nos indices das linhas não nos dados
                try:
                    treino_x, teste_x, treino_y, teste_y = train_test_split(X_indices, y, train_size=0.5, test_size=0.5, stratify=y)
                except ValueError:
                    print("This is related with the sample size, adjust later")
                df_treino_indices[str(iteracao+1)] = treino_x
                df_teste_indices[str(iteracao+1)] = teste_x
            
            df_treino_indices = pd.DataFrame(df_treino_indices, columns=["index", "value"])
            df_teste_indices = pd.DataFrame(df_teste_indices, columns=["index", "value"]) 
            df_treino_indices.to_csv(pasta_indices+"/treino-"+str(numero_repeticoes)+".csv")
            df_teste_indices.to_csv(pasta_indices+"/teste-"+str(numero_repeticoes)+".csv")
        
        
        #print("BASE: ",nome_base)
        ## cataloga todas as classes do conjunto de dados
        classes = [] # classes encontradas
        # adiciona classes que nao foram adicionadas, que estejam apenas no teste
        for classe in y:
            if classe not in classes:
                classes.append(classe)

        maximo = len(X.T) # numero maximo de features
        
        # para cada tecnica de redução de dimensionalidade
        for nome_reducao in nomes_reducao: # percorre as chaves
            #print("reducao: ",nome_reducao)

            # listas das medias para cada iteracao
            mediasTree = np.zeros(maximo)
            mediasNB = np.zeros(maximo)
            mediasKNN = np.zeros(maximo)
            mediasLD = np.zeros(maximo)
            
            # médias de todas as iteracoes pra cada base
            # cada feature tera uma lista de medias, para depois se obter o desvio padrão
            mediasIteracoesTree = []
            mediasIteracoesNB = []
            mediasIteracoesKNN = []
            mediasIteracoesLD = []
            
            pasta_salvar_pca = "bases/"+nome_base+"/autvetores_PCA_iteracoes/" 
            pasta_reducoes_resultados = "bases/"+nome_base+"/reducoes_resultados/"
            criar_pasta(pasta_salvar_pca) # cria a pasta onde ficara os pca de cada iteracao
            criar_pasta(pasta_reducoes_resultados)

            # usado para a quantidade de features media para certas reducoes
            quantidade_features_reducao = 0
            for iteracao in range(numero_repeticoes):
                print(nome_base, nome_reducao, iteracao)
                treino_indices = df_treino_indices[str(iteracao+1)]
                teste_indices = df_teste_indices[str(iteracao+1)]
                treino_x, teste_x = X[treino_indices], X[teste_indices]                
                treino_y, teste_y = y[treino_indices], y[teste_indices]
                
                autovetores = None
                autovalores = None
                treino_reduzido_x = None
                teste_reduzido_x = None
                
                ######### PCA ##############################
                
                # save e load do numpy nao funcionam como antes, entao autovetores serao recalculados
                pasta_iteracao_autvet = pasta_salvar_pca + "autvet-" + str(iteracao + 1)+".csv"
                pasta_iteracao_autval = pasta_salvar_pca + "autval-" + str(iteracao + 1)+".csv"
                if (not os.path.isfile(pasta_iteracao_autvet)): # se o pca não foi calculado
                    autovetores, autovalores = PCA(treino_x) # aplicacao do reducao normal
                    np.savetxt(pasta_iteracao_autvet, autovetores, delimiter=',')
                    np.savetxt(pasta_iteracao_autval, autovalores, delimiter=',')
                else: # se existem os dados sao carregados
                    autovetores = np.loadtxt(pasta_iteracao_autvet, delimiter=',')
                    autovalores = np.loadtxt(pasta_iteracao_autval, delimiter=',')

                # calculando autovetores e autovalores mesmo antes
                autovetores, autovalores = PCA(treino_x)  # aplicacao do reducao normal

                # Projecao do PCA
                projecao_treino_x = treino_x @ autovetores
                projecao_teste_x = teste_x @ autovetores
                ##############################################
                
                resultado_tree = []
                resultado_gnb = []
                resultado_knn = []
                resultado_lda = []
                
                ########## Demais Tipos de redução de dimensionalidade ########
                if "info_gain" in nome_reducao:
                    ordenado, feature_importances = info_gain(projecao_treino_x, treino_y, nomes_colunas)
                    feature_importances.to_csv(pasta_reducoes_resultados+"info_gain-"+str(iteracao+1)+".csv")
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:, ordenado]
                elif "fishers_score" in nome_reducao:
                    # indices das features ordenadas em ordem decrescente de acordo com o fisher score
                    ordenado, feature_importances = fishers_score(projecao_treino_x, treino_y, nomes_colunas)
                    feature_importances.to_csv(pasta_reducoes_resultados+"fisher_score-"+str(iteracao+1)+".csv")
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:, ordenado]
                elif "MCEPCA" in nome_reducao:
                    Sk = MCEPCA(projecao_treino_x, treino_x, maximo, classes, treino_y, autovalores, autovetores)
                    treino_reduzido_x = treino_x @ Sk
                    teste_reduzido_x = teste_x @ Sk
                elif "PCA" in nome_reducao:
                    treino_reduzido_x = projecao_treino_x
                    teste_reduzido_x = projecao_teste_x
                elif "chi2_square" in nome_reducao:
                    X_teste_cat = projecao_treino_x.astype(int) # passando para categorico
                    ordenado = chi2_square(X_teste_cat, treino_y, maximo)      
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:, ordenado]
                    
                elif "correlation_coefficient" in nome_reducao:
                    # problema: tem que fazer para todas as features e testar um bom 
                    # coeficiente de correlação (testar vários)
                    new_data = pd.DataFrame(data=projecao_treino_x, columns=[str(coluna) for coluna in range(maximo)]) # criação de um dataframe temporario para calcular a matriz de correlacao
                    ordenado = correlation_coefficient(new_data)
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:,ordenado]
                elif "RFE" in nome_reducao:
                    ordenado = RFE_linear_regression(projecao_treino_x, treino_y)
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:,ordenado]
                elif "Forward" in nome_reducao:
                    ordenado = forward_linear_regression(projecao_treino_x, treino_y)
                    treino_reduzido_x = projecao_treino_x[:, ordenado]
                    teste_reduzido_x = projecao_teste_x[:,ordenado]
                elif "variance_threshold" in nome_reducao:
                    # diferente dos outros acima, o variance treshold usa esta lista como definitiva, não reduz a quantidade
                    ind_features = variance_threshold(projecao_treino_x, 0.3)
                    print(ind_features)
                    quantidade_features_reducao += len(ind_features)
                    treino_reduzido_x = projecao_treino_x[:, ind_features]
                    teste_reduzido_x = projecao_teste_x[:, ind_features]
                elif "LASSO" in nome_reducao:
                    ind_features = LASSO(projecao_treino_x, treino_y)
                    print(ind_features)
                    quantidade_features_reducao += len(ind_features)
                    
                    treino_reduzido_x = projecao_treino_x[:, ind_features]
                    teste_reduzido_x = projecao_teste_x[:, ind_features]
                                        
                ###############################################################
                                
                # os resultados das excecoes sao feitos diferentes
                #if "correlation_coefficient" not in nome_reducao:
                    ############### Classificacao dos resultados ###############################
                
                if "variance_threshold" in nome_reducao or "LASSO" in nome_reducao:
                    quant_feat = len(X[0])
                    resultado_tree = list(np.full(quant_feat, treino_teste_unico(tree_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)))
                    resultado_gnb = list(np.full(quant_feat, treino_teste_unico(gnb_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)))
                    resultado_knn = list(np.full(quant_feat, treino_teste_unico(knn_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)))
                    resultado_lda = list(np.full(quant_feat, treino_teste_unico(lda_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)))
                else:
                    #print(treino_reduzido.shape)

                    ### reshape o array de treino de tree
                    #tree_reduzido_x = np.array([sample] for sample in treino_reduzido_x)
                    #print("Tree")
                    resultado_tree = treino_teste(tree_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)
                    #print("GNB")
                    resultado_gnb = treino_teste(gnb_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)
                    #print("KNN")
                    resultado_knn = treino_teste(knn_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)
                    #print("LDA")
                    resultado_lda = treino_teste(lda_reducao, treino_reduzido_x, teste_reduzido_x, treino_y, teste_y)

                ### adicionando resultados de cada feature
                mediasTree += resultado_tree
                mediasNB += resultado_gnb
                mediasKNN += resultado_knn
                mediasLD += resultado_lda

                ### matrizes usadas para desvio padrao
                mediasIteracoesTree.append(resultado_tree)
                mediasIteracoesNB.append(resultado_gnb)
                mediasIteracoesKNN.append(resultado_knn)
                mediasIteracoesLD.append(resultado_lda)
            
            # os vetores com as somas dos resultados em cada feature sao divididos para obter a media
            mediasTree = mediasTree/numero_repeticoes
            mediasNB = mediasNB/numero_repeticoes
            mediasKNN = mediasKNN/numero_repeticoes
            mediasLD = mediasLD/numero_repeticoes
            
            # desvio padrão
            desvioTree = []
            desvioNB=[]
            desvioKNN=[]
            desvioLD=[]
            # extraindo o desvio padrao
            for f in range(maximo):
                desvioTree.append(pd.Series(mediasTree).std())
                desvioNB.append(pd.Series(mediasNB).std())
                desvioKNN.append(pd.Series(mediasKNN).std())
                desvioLD.append(pd.Series(mediasLD).std())
        
            # salva os resultados em arquivos .txt
            salvar(mediasTree, mediasKNN, mediasNB, mediasLD, # resultados dos classificadores
                   ["resultados", "repeticoes-"+str(numero_repeticoes), nome_base], # nomes das pastas
                   nome_reducao, maximo) # nome do reducao

            # salva os desvios padroes em arquivos txt
            salvar(desvioTree, desvioKNN, desvioNB, desvioLD, # desvios padroes dos resultados anteriores
                   ["resultados", "repeticoes-"+str(numero_repeticoes), nome_base], # nomes das pastas
                   nome_reducao+"_desvio_padrao", maximo)
            
            if "variance_threshold" in nome_reducao or "LASSO" in nome_reducao:
                arquivo_reducoes = open("resultados/repeticoes-"+str(numero_repeticoes)+"/"+nome_base+"/quantidade-total-reducoes-"+nome_reducao+".txt", "w")
                arquivo_reducoes.writelines(str(quantidade_features_reducao))
                arquivo_reducoes.close()
                print("Média de features: "+str(int(quantidade_features_reducao/numero_repeticoes)))
                print("Original: "+str(len(X[0])))

        
