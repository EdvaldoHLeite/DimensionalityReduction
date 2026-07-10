from load_datasets import obs_network, letter, user_knowledge, mice, wine_quality_red, wine_quality_white, waveform
from load_datasets import banknote, climate, debrecen, occupancy, pima, vcolumn, wdbc, spambase
from projection import project_datasets
import warnings

# from plot_graphs import info_gain_graph, fisher_score_graph, graphs

warnings.filterwarnings("ignore")

def main():
    datasets1 = [
        #obs_network(),
        letter(),
        user_knowledge(),
        mice(),
        wine_quality_red(),
        wine_quality_white(),
        waveform()
    ]
    
    '''datasets2 = [
        banknote(),
        climate(),
        debrecen(),
        occupancy(),
        pima(),
        vcolumn(),
        wdbc(),
        #spambase() execute a later run
    ]'''
    datasets2 = [
        climate()
    ]
    
    repetition = 2

    ### correlation` coeficient is not in that previous tests (?)
    #reduction_names = ['RFE']
    reduction_names = [#'MCEPCA',
                    'PCA',
                     #'chi2_square',
                     #'LASSO',
                     #'fishers_score',
                     #'info_gain',
                     #'Forward',
                     #'RFE',
                     #'variance_threshold'
                     ]

    print("Dataset", len(datasets2))
    
    project_datasets(datasets2, reduction_names, repetition)



    '''import numpy as np
    for b in bases2:
        print(str(b[2]) + ": " + str(len(np.unique(b[1])))) # quantidade de y'''
main()
    























