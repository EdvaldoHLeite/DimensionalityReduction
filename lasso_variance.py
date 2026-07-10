from load_datasets import obs_network, letter, user_knowledge, mice, wine_quality_red, wine_quality_white, waveform
from load_datasets import banknote, climate, debrecen, occupancy, pima, vcolumn, wdbc, spambase
import pandas as pd


opt=3

bases3 = [
    letter()[2],
    user_knowledge()[2],
    mice()[2],
    wine_quality_red()[2],
    wine_quality_white()[2],
    waveform()[2],
]
reducoes3 = ['PCA', 'LASSO', 'variance_threshold']

bases2 = [
    banknote()[2],
    climate()[2],
    debrecen()[2],
    occupancy()[2],
    pima()[2],
    vcolumn()[2],
    wdbc()[2],
]
reducoes2 = ['MCEPCA', 'LASSO', 'variance_threshold']

bases = None
reducoes = None
classificadores = ["tree", 'knn','gnb','lda']
if (opt==2):    
    bases = bases2
    reducoes = reducoes2
else:
    bases = bases3
    reducoes = reducoes3

reducoes.reverse()
for b in bases:
    for r in reducoes:
        nome_arquivo = "resultados/repeticoes-100/" + b + "/" + r + ".txt" 
        f = pd.read_csv(nome_arquivo, sep=",")

        print(b, r)
        print(f[classificadores].mean())
        print('\n')


            
            
                
            
