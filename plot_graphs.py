# from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from load_datasets import *

# dataset names for plotting graphs
from load_datasets import (
    letter,
    mice,
    obs_network,
    user_knowledge,
    waveform,
    wine_quality_red,
    wine_quality_white,
)
from train_test import create_folder, load


def info_gain_graph(base, repeticoes):
    name = base[2]
    columns = base[3]

    # data frame
    feat_importances = pd.Series(data=np.arange(len(columns)), index=columns)
    # calculating the mean
    for i in range(repeticoes):
        data = pd.read_csv(
            "resultados/repeticoes-"
            + str(repeticoes)
            + "/"
            + name
            + "/reducoes_resultados/info_gain-"
            + str(i + 1)
            + ".csv"
        )

        data.columns = ["columns", "values"]
        dataSeries = pd.Series(
            data=np.array(data["values"]), index=np.array(data["columns"]).astype(str)
        )
        # print(dataSeries.index)
        for column in list(columns):
            # print(data[coluna])
            feat_importances[column] += dataSeries[column] / repeticoes
    # plotting
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind="barh", color="teal")
    plt.title(name + " - Info Gain")
    plt.show()
    plt.close()


def fisher_score_graph(base, repeticoes):
    name = base[2]
    columns = base[3]

    # data frame
    feat_importances = pd.Series(data=np.arange(len(columns)), index=columns)
    # calculating the mean
    for i in range(repeticoes):
        data = pd.read_csv(
            "resultados/repeticoes-"
            + str(repeticoes)
            + "/"
            + name
            + "/reducoes_resultados/fisher_score-"
            + str(i + 1)
            + ".csv"
        )

        data.columns = ["columns", "values"]
        dataSeries = pd.Series(
            data=np.array(data["values"]), index=np.array(data["columns"]).astype(str)
        )
        # print(dataSeries.index)
        for coluna in list(columns):
            # print(data[coluna])
            feat_importances[coluna] += dataSeries[coluna] / repeticoes
    # plotting
    feat_importances.sort_values(inplace=True)
    feat_importances.plot(kind="barh", color="teal")
    plt.title(name + " - Fisher`s Score")
    plt.show()
    plt.close()


def fisher_info_gain():
    bases = [
        banknote(),
        climate(),
        debrecen(),
        pima(),
        vcolumn(),
        wdbc(),
        spambase(),
        occupancy(),
    ]
    # bases=[debrecen()]
    for base in bases:
        fisher_score_graph(base, 100)
        info_gain_graph(base, 100)


def graphs():
    ################ PCAs #################################################################
    reduction_names = [
        "chi2_square",
        #'LASSO',
        "fishers_score",
        "info_gain",
        "Forward",
        "RFE",
        #'variance_threshold'
    ]

    """arranjos = []
    for reducao in nomes_reducao:
        arranjos.append(["PCA", reducao])"""

    main_pca = "MCEPCA"
    """arranjos = [
        [pcaPrincipal, 'Forward'],
        [pcaPrincipal, 'info_gain'],
        [pcaPrincipal, 'chi2_square']
        ]"""
    arrangements = [[main_pca, "Forward", "info_gain", "chi2_square"]]
    # print(arranjos)

    ############## configurations for graphical images (plots) ##############################
    config = {
        "PCA": ["PCA", "b", "."],
        "MCEPCA": ["MCEPCA", "b", "."],
        "info_gain": ["info_gain", "k", "+"],
        # "fishers_score":["fishers_score", 'g', '.'],
        # "correlation_coefficient":['correlation_coefficient', 'g', '.'],
        "chi2_square": ["chi2_square", "y", "*"],
        # "RFE":["RFE", 'r', '.'],
        "Forward": ["Forward", "r", "2"],
        # "variance_threshold":["Variance Threshold: 0.1", 'g', '.'],
        # "LASSO":["LASSO", "r", "."]
    }

    ###### classifiers
    classifiers = ["tree", "knn", "gnb", "lda"]

    ####### datasets
    bases2 = ["Banknote", "Climate", "Debrecen", "Pima", "VColumn", "WDBC", "Occupancy"]
    bases3 = [
        # obs_network(),
        letter()[2],
        user_knowledge()[2],
        mice()[2],
        wine_quality_red()[2],
        wine_quality_white()[2],
        waveform()[2],
    ]

    bases = bases2
    ##### running for the configurations
    for base in bases:
        for arrange in arrangements:
            load("resultados/repeticoes-100", base, classifiers, arrange, config)


graphs()
# fisher_info_gain()
