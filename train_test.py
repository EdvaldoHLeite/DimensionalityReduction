# executes classifier training and tracks accuracy for each feature. Saves results to a .txt file


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_folder(name):
    try:
        if not os.path.isdir(name):
            os.mkdir(name)
    except FileNotFoundError:
        print("Skiping for while")


# Vector with the results of each feature
def train_test(classifier, projected_train, projected_test, train_y, test_y):
    results = []
    for k in range(len(projected_train.T)):  # features
        train = projected_train[:, : k + 1]
        test = projected_test[:, : k + 1]

        train = train.astype(float)  # complex number in case
        # print("X",treino.shape)
        classifier.fit(train, train_y)

        test = test.astype(float)
        results.append(100 * classifier.score(test, test_y))

    return np.array(results)


# test to one feature
def unique_train_test(classifier, projected_train, projected_test, train_y, test_y):
    train = projected_train.astype(float)
    test = projected_test.astype(float)

    classifier.fit(train, train_y)

    return 100 * classifier.score(test, test_y)


# tree, knn, gnb, and lda; holds results for each dataset, and each dataset tracks performance per feature
def save(tree, knn, gnb, lda, folders, pca_name, feature_count):

    ## files
    path = ""
    for folder in folders:
        path = path + folder + "/"
        create_folder(path)  # creates the folder tree

    file = open(path + pca_name + ".txt", "w")

    # headers
    results = "feature,tree,knn,gnb,lda\n"
    for f in range(feature_count):  # loops through the number of features
        results += str(f + 1) + ","
        results += str(tree[f]) + ","
        results += str(knn[f]) + ","
        results += str(gnb[f]) + ","
        results += str(lda[f])
        results += "\n"

    # saves to file
    file.writelines(results)

    file.close()


# names of the four classifiers, matching the layout order of the subplots
def load(
    folder, base_name, classifier_names, pca_names, config
):  # config is a dictionary containing label, color, and marker for each pca
    path = folder + "/" + base_name
    # print(nomesPCA)
    # minimum and maximum values for the y-axis scale
    ymin = 10000000
    ymax = 0

    classifiers = {
        "tree": "Decision Tree",
        "knn": "1-Nearest Neighbor",
        "gnb": "Naive Bayes",
        "lda": "Linear Discriminant",
    }
    ax = plt.subplot(2, 2, 1, title=classifiers[classifier_names[0]])
    for i in range(len(classifier_names)):
        # dados = pd.read_csv(caminho+'/'+nomes_classificadores[i]+'.txt')
        # creates a subplot for the classifier

        if i > 0:  # plotting on the same scale
            ax = plt.subplot(
                2,
                2,
                i + 1,
                title=classifiers[
                    classifier_names[i]
                ],  # sets graph title to the classifier name
                sharex=ax,
                sharey=ax,
            )

        # plots the performance curve for each pca method
        for pca_name in pca_names:
            datas = pd.read_csv(path + "/" + pca_name + ".txt")
            plt.plot(
                datas["feature"],  # x
                datas[classifier_names[i]],  # y
                label=config[pca_name][0],
                color=config[pca_name][1],
                marker=config[pca_name][2],
            )
            plt.grid(b=True)

            # scale can only be explicitly ticks-formatted if the number of features is relatively small
            if max(datas["feature"]) < 22:
                plt.xticks(
                    datas["feature"]
                )  # forces identical x-axis ticks across plots

            # updating y-axis scale boundaries
            if min(datas[classifier_names[i]]) < ymin:
                ymin = min(datas[classifier_names[i]])
            if max(datas[classifier_names[i]]) > ymax:
                ymax = max(datas[classifier_names[i]])

    # y scale execution
    y_scale = range(int(ymin), int(ymax))
    if len(y_scale) < 20:
        plt.yticks(range(int(ymin), int(ymax) + 2))

    plt.suptitle(base_name, fontsize=16)
    # positioning the legend box
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(-0.4, -0.09),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    # full screen handling
    full = plt.get_current_fig_manager()
    full.full_screen_toggle()

    plt.show()

    """nomePng = nomeBase+"-"
    for n in nomesPCA:
        nomePng += n    
    plt.savefig("graficos/"+nomePng+".png", bbox_inches='tight')"""
