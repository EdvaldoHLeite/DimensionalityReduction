# executa o treinamento e a acuracia dos classificadores, para cada feature. Salva os resultados em txt


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import os

def criar_pasta(nome):
    if not os.path.isdir(nome):
        os.mkdir(nome)

# vetor com resultado de cada feature
def treino_teste(classificador, treino_projetado, teste_projetado, treino_y, teste_y):
    resultados = []
    for k in range(len(treino_projetado.T)): # features
        treino = treino_projetado[:, :k+1]
        teste = teste_projetado[:, :k+1]
        
        treino = treino.astype(float) # caso esteja em complexo
        #print("X",treino.shape)
        classificador.fit(treino, treino_y)

        teste = teste.astype(float)
        resultados.append(100*classificador.score(teste, teste_y))

    return np.array(resultados)

# Faz o teste para uma unica quantidade de features
def treino_teste_unico(classificador, treino_projetado, teste_projetado, treino_y, teste_y):
    treino = treino_projetado.astype(float)
    teste = teste_projetado.astype(float)
    
    classificador.fit(treino, treino_y)
    
    return 100*classificador.score(teste, teste_y)

# tree, knn, gnb e lda; tem os resultados de cada base e cada base tem os resultados de cada feature
def salvar(tree, knn, gnb, lda, pastas, nome_pca, num_features):
    
    ## arquivos
    caminho = ""
    for pasta in pastas:
        caminho = caminho + pasta + "/"
        criar_pasta(caminho) # cria a arvore de pastas

    arquivo = open(caminho + nome_pca + ".txt", "w")

    # cabecalhos
    resultados = "feature,tree,knn,gnb,lda\n"
    for f in range(num_features): # percorre a quantidade de features
        resultados += str(f+1) + ","
        resultados += str(tree[f]) + ","
        resultados += str(knn[f]) + ","
        resultados += str(gnb[f]) + ","
        resultados += str(lda[f])
        resultados += '\n'

    # salva no arquivo
    arquivo.writelines(resultados)

    arquivo.close()
            
# nomes dos quatro classificadores, a ordem eh a mesma da plotagem dos subgraficos
def carregar(pasta, nomeBase, nomes_classificadores, nomesPCA, config): # config eh um dicionario com label, cor e marker para cada pca
    caminho = pasta + '/' + nomeBase
    #print(nomesPCA)
    # valores minimo e maximo para a escala em y
    ymin = 10000000
    ymax = 0

    classificadores = {'tree':"Decision Tree", 'knn':"1-Nearest Neighbor", 'gnb':"Naive Bayes", 'lda':"Linear Discriminant"}
    ax = plt.subplot(2, 2, 1, title=classificadores[nomes_classificadores[0]])
    for i in range(len(nomes_classificadores)):
        # dados = pd.read_csv(caminho+'/'+nomes_classificadores[i]+'.txt')
        # cria um subgrafico do classificador
        
        if i > 0: # plotagem na mesma escala
            ax = plt.subplot(2, 2, i+1,
                             title=classificadores[nomes_classificadores[i]], # nome do grafico sendo o do classificador
                             sharex=ax,
                             sharey=ax) 
        
        # plota o grafico de cada pca
        for nome_pca in nomesPCA:
            dados = pd.read_csv(caminho+'/'+nome_pca+'.txt')
            plt.plot(dados['feature'], # x
                     dados[nomes_classificadores[i]], # y
                     label=config[nome_pca][0],
                     color=config[nome_pca][1],
                     marker=config[nome_pca][2])
            plt.grid(b=True)

            # a escala so pode ser aplicada se no grafico nao tiver um grande numero de instancias
            if max(dados['feature']) < 22:
                plt.xticks(dados['feature']) # cada grafico com a mesma escala em x

            # atualizando limites da escala de y
            if min(dados[nomes_classificadores[i]]) < ymin: 
                ymin = min(dados[nomes_classificadores[i]])
            if max(dados[nomes_classificadores[i]]) > ymax:
                ymax = max(dados[nomes_classificadores[i]])

    # escala de y
    escala_y = range(int(ymin), int(ymax))
    if len(escala_y) < 20:
        plt.yticks(range(int(ymin), int(ymax)+2))

    plt.suptitle(nomeBase, fontsize=16)
    # posicionando a legenda
    plt.legend(loc='upper center', bbox_to_anchor=(-0.4, -0.09),fancybox=True, shadow=True, ncol=5)
    # full
    full = plt.get_current_fig_manager()
    full.full_screen_toggle()
    
    plt.show()

    '''nomePng = nomeBase+"-"
    for n in nomesPCA:
        nomePng += n    
    plt.savefig("graficos/"+nomePng+".png", bbox_inches='tight')'''
